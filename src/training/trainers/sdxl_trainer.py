"""SDXL trainer implementation with routing to specific training methods."""
import torch
from typing import Dict, Optional, Any, Literal
from pathlib import Path

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.core.distributed import is_main_process
from .base import BaseTrainer
from .ddpm_trainer import DDPMTrainer
from .flow_matching_trainer import FlowMatchingTrainer

logger = get_logger(__name__)

class SDXLTrainer(BaseTrainer):
    """SDXL trainer with routing to specific training methods."""
    
    AVAILABLE_METHODS = Literal["ddpm", "flow_matching"]
    
    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        training_method: AVAILABLE_METHODS = "ddpm",
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize SDXL trainer and route to specific training method.
        
        Args:
            model: SDXL model instance
            optimizer: Optimizer instance
            train_dataloader: Training data loader
            training_method: Training method to use ("ddpm" or "flow_matching")
            device: Torch device to use
            **kwargs: Additional arguments passed to specific trainer
        """
        super().__init__(model, optimizer, train_dataloader, device, **kwargs)
        
        # Initialize specific trainer based on method
        self.training_method = training_method
        if training_method == "ddpm":
            self.method_trainer = DDPMTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                **kwargs
            )
        elif training_method == "flow_matching":
            self.method_trainer = FlowMatchingTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown training method: {training_method}. "
                f"Available methods: {list(self.AVAILABLE_METHODS.__args__)}"
            )
            
        logger.info(f"Initialized SDXL trainer with {training_method} method")
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Route training step to specific method trainer."""
        return self.method_trainer.training_step(batch)
        
    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint with training method info."""
        kwargs["training_method"] = self.training_method
        super().save_checkpoint(path, **kwargs)
        
    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        **kwargs
    ) -> "SDXLTrainer":
        """Load checkpoint and initialize correct training method.
        
        Args:
            path: Path to checkpoint
            model: SDXL model instance
            optimizer: Optimizer instance
            train_dataloader: Training data loader
            **kwargs: Additional arguments
            
        Returns:
            Initialized trainer with correct method
        """
        checkpoint = torch.load(path, map_location=kwargs.get("device", "cuda"))
        training_method = checkpoint.get("training_method", "ddpm")
        
        trainer = cls(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            training_method=training_method,
            **kwargs
        )
        
        trainer.load_checkpoint(path)
        return trainer
        
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for SDXL model input.
        
        This method handles common SDXL-specific batch preparation
        that's needed regardless of training method.
        """
        # Get original image size and aspect ratio
        height, width = batch["pixel_values"].shape[-2:]
        original_size = (height, width)
        target_size = (height, width)  # Can be different for random resizing
        crops_coords_top_left = (0, 0)  # Can be random for crop augmentation
        
        # Create time_ids tensor for SDXL
        time_ids = torch.tensor([
            list(original_size) + list(target_size) + list(crops_coords_top_left) + [width/height]
        ], device=self.device)
        time_ids = time_ids.repeat(batch["pixel_values"].shape[0], 1)
        
        # Add to batch
        batch["time_ids"] = time_ids
        batch["original_size"] = original_size
        batch["target_size"] = target_size
        batch["crops_coords_top_left"] = crops_coords_top_left
        
        return batch
        
    def train_epoch(self) -> Dict[str, float]:
        """Override train_epoch to add SDXL-specific batch preparation."""
        self.model.train()
        epoch_metrics = {}
        
        for batch in self.train_dataloader:
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Prepare batch with SDXL-specific processing
                batch = self.prepare_batch(batch)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass through specific trainer
                outputs = self.training_step(batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                metrics = outputs.get("metrics", {})
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v)
                    
                self.step += 1
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}", exc_info=True)
                continue
                
        # Compute epoch metrics
        epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        
        # Log metrics
        if self.wandb_logger and is_main_process():
            self.wandb_logger.log_metrics({
                f"epoch_{k}": v for k, v in epoch_metrics.items()
            })
            
        return epoch_metrics
