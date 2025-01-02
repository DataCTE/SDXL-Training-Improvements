from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.core.logging import WandbLogger
from src.data.config import Config
from src.training.trainers.base_router import BaseRouter
from src.core.logging import get_logger
import os
from pathlib import Path

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
    from pathlib import Path
    
    # Convert base path
    base_path = "final_checkpoint" if is_final else f"checkpoint_{epoch}"
    path = convert_windows_path(base_path)
    save_path = Path(path)
    
    try:
        # Save model weights in safetensors format
        logger.info(f"Saving model checkpoint to {save_path}")
        model.save_pretrained(
            str(save_path),
            safe_serialization=True
        )
        
        # Save optimizer state
        optimizer_path = save_path / "optimizer.pt"
        logger.info(f"Saving optimizer state to {optimizer_path}")
        torch.save(
            optimizer.state_dict(),
            str(optimizer_path),
            _use_new_zipfile_serialization=False
        )
        
        # Save config
        config_path = save_path / "config.json"
        logger.info(f"Saving training config to {config_path}")
        config.save(str(config_path))
        
        logger.info(f"Successfully saved checkpoint to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
        raise
        
