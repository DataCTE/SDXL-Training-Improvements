"""SDXL trainer wrapper for backward compatibility."""
from typing import List, Optional, Union

import torch
from src.core.logging import WandbLogger
from src.core.history import TorchHistory
from src.data.config import Config
from src.models.sdxl import StableDiffusionXLModel, StableDiffusionXLPipeline
from src.training.trainers.sdxl_trainer import SDXLTrainer

def create_trainer(
    config: Config,
    model: StableDiffusionXLModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,  # Ensure num_workers=0 in DataLoader initialization
    device: Union[str, torch.device],
    wandb_logger: Optional[WandbLogger] = None,
    validation_prompts: Optional[List[str]] = None
) -> SDXLTrainer:
    """Wrapper around SDXLTrainer.create for backward compatibility."""
    return SDXLTrainer.create(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        device=device,
        wandb_logger=wandb_logger,
        validation_prompts=validation_prompts
    )
