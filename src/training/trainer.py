"""SDXL trainer implementation."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from ..core.types import DataType, ModelWeightDtypes

from ..core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_gc
)
from ..core.types import DataType, ModelWeightDtypes
from ..data.config import Config
from ..core.distributed import is_main_process, get_world_size
from ..core.logging import WandbLogger, log_metrics
from ..models import StableDiffusionXLModel
from .noise import generate_noise, get_add_time_ids

logger = logging.getLogger(__name__)

class SDXLTrainer:
    """Trainer for SDXL fine-tuning with advanced features."""
    
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            accelerator: Accelerator for distributed training
            unet: UNet model
            optimizer: Optimizer
            scheduler: Noise scheduler
            train_dataloader: Training data loader
            device: Target device
        """
        self.config = config
        self.model = model
        self.unet = model.unet  # Extract UNet from SDXL model
        self.optimizer = optimizer
        self.noise_scheduler = scheduler
        self.train_dataloader = train_dataloader
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
        
        # Move model and optimizer to device efficiently
        if not tensors_match_device(self.model.state_dict(), device):
            with create_stream_context(torch.cuda.current_stream()):
                tensors_to_device_(self.model.state_dict(), device, non_blocking=True)
                if hasattr(self.optimizer, 'state'):
                    tensors_to_device_(self.optimizer.state, device, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            torch_gc()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.max_steps = config.training.max_train_steps or (
            len(train_dataloader) * config.training.num_epochs
        )
        
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ):
        """Initialize trainer."""
        super().__init__()
        self.config = config
        self.model = model
        self.unet = model.unet
        self.optimizer = optimizer
        self.noise_scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.device = device
        self.wandb_logger = wandb_logger
        
        # Initialize training method
        if config.training.method == "flow_matching":
            from .methods.flow_matching import FlowMatchingMethod
            self.training_method = FlowMatchingMethod()
        else:
            from .methods.ddpm import DDPMMethod
            self.training_method = DDPMMethod(
                prediction_type=config.training.prediction_type
            )
            
        # Rest of initialization...

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, float]:
        """Execute single training step."""
        # Compute loss using selected method
        loss_dict = self.training_method.compute_loss(
            self.unet,
            batch,
            self.noise_scheduler,
            generator
        )
        
        # Backpropagate
        loss_dict["loss"].backward()
        
        # Gradient clipping
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
                
            # Training step
            step_metrics = self.train_step(batch)
            
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
        logger.info("Starting training...")
        
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
