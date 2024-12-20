"""SDXL trainer implementation with factory support for multiple training methods."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core.distributed import is_main_process, get_world_size
from src.core.logging import WandbLogger, log_metrics
from src.core.memory import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_gc,
    setup_memory_optimizations,
    verify_memory_optimizations,
    LayerOffloader,
    LayerOffloadConfig,
    ThroughputMonitor
)
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.models import StableDiffusionXLModel
from src.training.methods.base import BaseTrainingMethod

logger = logging.getLogger(__name__)

class SDXLTrainer:
    """Base trainer class for SDXL supporting multiple training methods."""
    
    @classmethod
    def create(
        cls,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ) -> 'SDXLTrainer':
        """Factory method to create appropriate trainer instance.
        
        Args:
            config: Training configuration
            model: SDXL model
            optimizer: Optimizer
            train_dataloader: Training data loader
            device: Target device
            wandb_logger: Optional W&B logger
            validation_prompts: Optional validation prompts
            
        Returns:
            Configured trainer instance
        """
        # Get trainer class using metaclass registry
        method = config.training.method.lower()
        trainer_cls = BaseTrainingMethod.get_method(method)
        logger.info(f"Creating trainer with method: {trainer_cls.__name__}")
        
        # Create training method instance
        training_method = trainer_cls(
            unet=model.unet,
            config=config
        )
        
        # Create and return trainer
        return cls(
            config=config,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            training_method=training_method,
            device=device,
            wandb_logger=wandb_logger,
            validation_prompts=validation_prompts
        )
    
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        training_method: BaseTrainingMethod,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ):
        """Initialize base trainer.
        
        Args:
            config: Training configuration
            model: SDXL model
            optimizer: Optimizer
            train_dataloader: Training data loader
            training_method: Training method implementation
            device: Target device
            wandb_logger: Optional W&B logger
            validation_prompts: Optional validation prompts
        """
        self.config = config
        self.model = model
        self.unet = model.unet
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.training_method = training_method
        self.device = device
        self.wandb_logger = wandb_logger
        
        # Configure model dtypes
        self.model_dtypes = ModelWeightDtypes(
            train_dtype=DataType.FLOAT_32,
            fallback_train_dtype=DataType.FLOAT_16,
            unet=DataType.FLOAT_32,
            prior=DataType.FLOAT_32,
            text_encoder=DataType.FLOAT_32,
            text_encoder_2=DataType.FLOAT_32,
            text_encoder_3=DataType.FLOAT_32,
            vae=DataType.FLOAT_32,
            effnet_encoder=DataType.FLOAT_32,
            decoder=DataType.FLOAT_32,
            decoder_text_encoder=DataType.FLOAT_32,
            decoder_vqgan=DataType.FLOAT_32,
            lora=DataType.FLOAT_32,
            embedding=DataType.FLOAT_32
        )
        
        # Initialize memory management
        self._setup_memory_management(
            batch_size=train_dataloader.batch_size,
            micro_batch_size=config.training.micro_batch_size
        )
        
        # Move model to device with proper memory handling
        if not tensors_match_device(self.model.state_dict(), device):
            with create_stream_context(torch.cuda.current_stream()):
                tensors_to_device_(self.model.state_dict(), device)
                if hasattr(self.optimizer, 'state'):
                    tensors_to_device_(self.optimizer.state, device)
            torch.cuda.current_stream().synchronize()
        torch_gc()
            
        # Training state and monitoring
        self.global_step = 0
        self.epoch = 0
        self.max_steps = config.training.max_train_steps or (
            len(train_dataloader) * config.training.num_epochs
        )
        self.throughput_monitor = ThroughputMonitor()
        
        # Setup gradient accumulation
        self.gradient_accumulation_steps = (
            train_dataloader.batch_size // config.training.micro_batch_size 
            if config.training.micro_batch_size else 1
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        """Execute single training step.
        
        Args:
            batch: Training batch
            generator: Optional random generator
            
        Returns:
            Dict of metrics
        """
        # Compute loss using selected method
        loss_dict = self.training_method.compute_loss(
            self.unet,
            batch,
            generator=generator
        )
        
        # Scale loss for gradient accumulation
        loss = loss_dict["loss"] / self.gradient_accumulation_steps
        loss.backward()
        
        # Only update on last accumulation step
        if accumulation_step == self.gradient_accumulation_steps - 1:
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.unet.parameters(),
                    self.config.training.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {k: v.detach().item() for k, v in loss_dict.items()}
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict of epoch metrics
        """
        epoch_metrics = {}
        
        # Set models to training mode
        self.model.train()
        
        # Training loop
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            disable=not is_main_process(),
            desc=f"Epoch {self.epoch}"
        )
        
        for batch in self.train_dataloader:
            # Skip if max steps reached
            if self.global_step >= self.max_steps:
                break
                
            # Process micro-batches
            try:
                micro_batches = self._prepare_micro_batches(batch)
                step_metrics = {}
                
                for i, micro_batch in enumerate(micro_batches):
                    metrics = self.train_step(micro_batch, accumulation_step=i)
                    step_metrics.update(metrics)
                
                # Update throughput monitoring
                self.throughput_monitor.update(batch["model_input"].shape[0])
                step_metrics.update(self.throughput_monitor.get_metrics())
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU OOM at step {self.global_step}. Attempting recovery...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Error during training step: {str(e)}")
                    raise
            
            # Update metrics
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
                
            # Log metrics
            if self.global_step % self.config.training.log_steps == 0:
                log_metrics(
                    step_metrics,
                    self.global_step,
                    is_main_process=is_main_process(),
                    use_wandb=self.wandb_logger is not None,
                    wandb_logger=self.wandb_logger
                )
                
            # Save checkpoint
            if (
                self.config.training.save_steps > 0 and
                self.global_step % self.config.training.save_steps == 0
            ):
                self.save_checkpoint()
                
            self.global_step += 1
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Compute epoch metrics
        epoch_metrics = {
            k: sum(v) / len(v)
            for k, v in epoch_metrics.items()
        }
        
        return epoch_metrics
        
    def train(self) -> Dict[str, float]:
        """Execute complete training loop.
        
        Returns:
            Dict of final metrics
        """
        logger.info(f"Starting training with {self.training_method.name} method...")
        
        metrics = {}
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            log_metrics(
                epoch_metrics,
                epoch,
                is_main_process=is_main_process(),
                use_wandb=self.config.training.use_wandb,
                step_type="epoch"
            )
            
            # Update metrics
            metrics.update(epoch_metrics)
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Check if max steps reached
            if self.global_step >= self.max_steps:
                break
                
        return metrics
        
    def _prepare_micro_batches(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Split batch into micro-batches for gradient accumulation."""
        if self.gradient_accumulation_steps == 1:
            return [batch]
            
        micro_batches = []
        batch_size = batch["model_input"].shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        
        for i in range(self.gradient_accumulation_steps):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            
            micro_batch = {
                k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            micro_batches.append(micro_batch)
            
        return micro_batches

    def _setup_memory_management(
        self,
        batch_size: Optional[int] = None,
        micro_batch_size: Optional[int] = None
    ) -> None:
        """Initialize memory optimizations and management.
        
        Args:
            batch_size: Training batch size
            micro_batch_size: Micro batch size for gradient accumulation
        """
        # Setup core memory optimizations
        self.memory_optimized = setup_memory_optimizations(
            model=self.model,
            config=self.config,
            device=self.device,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size
        )
        
        if self.memory_optimized:
            # Verify optimizations are active
            verify_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=logger
            )
            
            # Configure layer offloading if enabled
            if self.config.training.memory.enable_24gb_optimizations:
                self.layer_offloader = LayerOffloader(
                    model=self.model,
                    config=LayerOffloadConfig(
                        enabled=True,
                        fraction=self.config.training.memory.layer_offload_fraction,
                        temp_device=self.config.training.memory.temp_device,
                        async_transfer=self.config.training.memory.enable_async_offloading
                    ),
                    device=self.device
                )
                
        # Set up tensor cleanup hooks
        def cleanup_hook():
            torch_gc()
        self.cleanup_hook = cleanup_hook

    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        if not is_main_process():
            return
            
        logger.info(f"Saving checkpoint at step {self.global_step}...")
        
        checkpoint_dir = Path(self.config.global_config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch
            },
            checkpoint_dir / "state.pt"
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
