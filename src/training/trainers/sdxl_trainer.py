from typing import Optional
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.core.logging import WandbLogger, get_logger
from src.data.config import Config
from src.training.trainers.base_router import BaseTrainer
from src.core.distributed import is_main_process

logger = get_logger(__name__)

class SDXLTrainer(BaseTrainer):
    """SDXL-specific trainer that handles model saving and training method delegation."""
    
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
        
        # Initialize specific training method through composition
        if config.training.method.lower() == "ddpm":
            from .methods.ddpm_trainer import DDPMTrainer
            self.method_trainer = DDPMTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                **kwargs
            )
        elif config.training.method.lower() == "flow_matching":
            from .methods.flow_matching_trainer import FlowMatchingTrainer
            self.trainer = FlowMatchingTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported training method: {config.training.method}")
    
    def train(self, num_epochs: int):
        """Delegate training to the specific trainer implementation."""
        if not hasattr(self, 'trainer'):
            # If this is a subclass (like DDPMTrainer), execute its own train method
            return super().train(num_epochs)
        
        # Otherwise delegate to the specific trainer
        logger.info(f"Delegating training to {self.trainer.__class__.__name__}")
        return self.trainer.train(num_epochs)
    
    def save_checkpoint(self, save_path: Path, is_final: bool = False):
        """Save checkpoint in diffusers format using save_pretrained with safetensors."""
        if not is_main_process():
            return
            
        from src.data.utils.paths import convert_windows_path
        
        # Convert base path
        base_path = "final_checkpoint" if is_final else str(save_path)
        path = convert_windows_path(base_path)
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model weights in safetensors format
            logger.info(f"Saving model checkpoint to {save_path}")
            self.model.save_pretrained(
                str(save_path),
                safe_serialization=True
            )
            
            # Save optimizer state
            optimizer_path = save_path / "optimizer.pt"
            logger.info(f"Saving optimizer state to {optimizer_path}")
            torch.save(
                self.optimizer.state_dict(),
                str(optimizer_path),
                _use_new_zipfile_serialization=False
            )
            
            # Save config as JSON
            config_path = save_path / "config.json"
            logger.info(f"Saving training config to {config_path}")
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info(f"Successfully saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise
        
