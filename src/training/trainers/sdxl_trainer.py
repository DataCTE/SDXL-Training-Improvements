from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.core.logging import WandbLogger
from src.data.config import Config
from src.training.trainers.base_router import BaseRouter
from src.core.logging import get_logger
import os

logger = get_logger(__name__)

class SDXLTrainer(BaseRouter):
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
        wandb_logger: Optional[WandbLogger] = None,
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
        
        # Get gradient accumulation steps from config
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        
        # Verify batch size matches config
        if config and hasattr(train_dataloader, 'batch_size'):
            if train_dataloader.batch_size != config.training.batch_size:
                logger.warning(
                    f"DataLoader batch size ({train_dataloader.batch_size}) "
                    f"doesn't match config batch size ({config.training.batch_size})"
                )
        
    def train(self, num_epochs: int):
        """Delegate training to the specific trainer implementation."""
        if not hasattr(self, 'trainer'):
            # If this is a subclass (like DDPMTrainer), execute its own train method
            return super().train(num_epochs)
        
        # Otherwise delegate to the specific trainer
        logger.info(f"Delegating training to {self.trainer.__class__.__name__}")
        return self.trainer.train(num_epochs)
    
    
def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    config: Config,
    is_final: bool = False
):
    """Save checkpoint in diffusers format using save_pretrained with safetensors."""
    from src.data.utils.paths import convert_windows_path
    
    # Convert base path
    base_path = "final_checkpoint" if is_final else f"checkpoint_{epoch}"
    path = convert_windows_path(base_path)
    
    # Save model weights in safetensors format
    model.save_pretrained(
        str(path),  # convert Path to string for save_pretrained
        safe_serialization=True
    )
    
    # Save optimizer state with converted path
    optimizer_path = convert_windows_path(os.path.join(str(path), "optimizer.safetensors"))
    torch.save(
        optimizer.state_dict(),
        str(optimizer_path),
        _use_new_zipfile_serialization=False
    )
    
    # Save config with converted path
    config_path = convert_windows_path(os.path.join(str(path), "config.json"))
    config.save(str(config_path))
        
