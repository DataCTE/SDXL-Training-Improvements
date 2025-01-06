"""Template for implementing new training methods."""
import torch
from typing import Dict, Optional, Any
from torch import Tensor
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import time

from src.core.logging import UnifiedLogger, LogConfig
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config

logger = UnifiedLogger(LogConfig(name=__name__))

class ExampleMethodTrainer(SDXLTrainer):
    """
    Template trainer class for new training methods.
    
    Steps to implement a new method:
    1. Rename this class to YourMethodTrainer
    2. Set the name class variable to your method's name
    3. Implement the compute_loss method with your training logic
    4. Optionally override other methods if needed
    """
    
    name = "example_method"  # Change this to your method's name

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        wandb_logger=None,
        config: Optional[Config] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=device,
            wandb_logger=wandb_logger,
            config=config,
            **kwargs
        )
        
        # Standard initialization from base trainer
        self.effective_batch_size = (
            config.training.batch_size * 
            self.gradient_accumulation_steps
        )
        
        # Setup memory optimizations
        self._setup_memory_optimizations(config)
        
        # Setup mixed precision training
        self._setup_mixed_precision(config)
        
        # Setup model dtype
        self.model_dtype = self._setup_model_dtype(config)
        
    def _setup_memory_optimizations(self, config: Config) -> None:
        """Setup memory optimizations for training."""
        # Gradient checkpointing
        if hasattr(self.model.unet, "enable_gradient_checkpointing"):
            logger.info("Enabling gradient checkpointing for UNet")
            self.model.unet.enable_gradient_checkpointing()
        
        if hasattr(self.model.vae, "enable_gradient_checkpointing"):
            logger.info("Enabling gradient checkpointing for VAE")
            self.model.vae.enable_gradient_checkpointing()
        
        # xFormers optimizations
        if config.training.enable_xformers:
            logger.info("Enabling xformers memory efficient attention")
            if hasattr(self.model.unet, "enable_xformers_memory_efficient_attention"):
                self.model.unet.enable_xformers_memory_efficient_attention()
            if hasattr(self.model.vae, "enable_xformers_memory_efficient_attention"):
                self.model.vae.enable_xformers_memory_efficient_attention()
        
        # CPU offload
        if hasattr(self.model, "enable_model_cpu_offload"):
            logger.info("Enabling model CPU offload")
            self.model.enable_model_cpu_offload()

    def _setup_mixed_precision(self, config: Config) -> None:
        """Setup mixed precision training."""
        self.mixed_precision = config.training.mixed_precision
        if self.mixed_precision != "no":
            self.scaler = GradScaler()

    def _setup_model_dtype(self, config: Config) -> torch.dtype:
        """Setup model data types."""
        if config.optimizer.optimizer_type == "adamw_bf16":
            logger.info("Converting model to bfloat16")
            bfloat16_weights = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
            self.model.unet.to(bfloat16_weights.unet.to_torch_dtype())
            self.model.vae.to(bfloat16_weights.vae.to_torch_dtype())
            self.model.text_encoder_1.to(bfloat16_weights.text_encoder.to_torch_dtype())
            self.model.text_encoder_2.to(bfloat16_weights.text_encoder_2.to_torch_dtype())
            return torch.bfloat16
        return next(self.model.parameters()).dtype

    def compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Implement your method's loss computation here.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary containing:
                - loss: The computed loss tensor
                - metrics: Dictionary of metrics to log
        """
        raise NotImplementedError(
            "Implement compute_loss() for your training method"
        )

    def _execute_training_step(
        self,
        batch: Dict[str, Tensor],
        accumulate: bool = False,
        is_last_accumulation_step: bool = True
    ) -> tuple[Tensor, dict]:
        """Execute single training step with gradient accumulation support."""
        try:
            if not accumulate or is_last_accumulation_step:
                self.optimizer.zero_grad()
            
            step_output = self.compute_loss(batch)
            loss = step_output["loss"]
            metrics = step_output["metrics"]
            
            if accumulate:
                loss = loss / self.config.training.gradient_accumulation_steps
            
            loss.backward()
            
            return loss * self.config.training.gradient_accumulation_steps, metrics
            
        except Exception as e:
            logger.error(f"{self.name} training step failed", exc_info=True)
            raise

    def train(self, num_epochs: int) -> None:
        """Standard training loop implementation."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting {self.name} training: {total_steps} steps ({num_epochs} epochs)")
        
        global_step = 0
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc=f"Training {self.name}",
            position=0,
            leave=True
        )
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                self.model.train()
                
                accumulated_loss = 0.0
                accumulated_metrics = defaultdict(float)
                
                for step, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    loss, metrics = self._execute_training_step(
                        batch, 
                        accumulate=True,
                        is_last_accumulation_step=((step + 1) % self.gradient_accumulation_steps == 0)
                    )
                    
                    step_time = time.time() - step_start_time
                    progress_bar.set_postfix(
                        {'Loss': f"{loss.item():.4f}", 'Time': f"{step_time:.1f}s"},
                        refresh=True
                    )
                    
                    accumulated_loss += loss.item()
                    for k, v in metrics.items():
                        accumulated_metrics[k] += v
                        
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        effective_loss = accumulated_loss / self.gradient_accumulation_steps
                        effective_metrics = {
                            k: v / self.gradient_accumulation_steps 
                            for k, v in accumulated_metrics.items()
                        }
                        
                        if self.config.training.clip_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.training.clip_grad_norm
                            )
                            effective_metrics['grad_norm'] = grad_norm.item()
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        accumulated_loss = 0.0
                        accumulated_metrics.clear()
                        
                        if is_main_process() and self.wandb_logger:
                            self.wandb_logger.log_metrics(
                                {
                                    'epoch': epoch + 1,
                                    'step': global_step,
                                    'loss': effective_loss,
                                    'step_time': step_time,
                                    **effective_metrics
                                },
                                step=global_step
                            )
                    
                    global_step += 1
                    progress_bar.update(1)
                
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch + 1}, step {step + 1}", exc_info=True)
            raise
        finally:
            progress_bar.close()
