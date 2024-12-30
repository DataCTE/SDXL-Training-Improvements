from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.core.logging import WandbLogger
from src.data.config import Config
from src.training.trainers.base_router import BaseRouter

class SDXLTrainer(BaseRouter):
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: DataLoader,
        training_method: str,
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
        
        # Import trainers here to avoid circular imports
        from src.training.trainers.methods.ddpm_trainer import DDPMTrainer
        from src.training.trainers.methods.flow_matching_trainer import FlowMatchingTrainer
        
        # Create appropriate training method
        if training_method == "ddpm":
            self.trainer = DDPMTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                **kwargs
            )
        elif training_method == "flow_matching":
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
            raise ValueError(f"Unknown training method: {training_method}")

    def train(self, num_epochs: int):
        """Delegate training to the specific trainer implementation."""
        return self.trainer.train(num_epochs) 