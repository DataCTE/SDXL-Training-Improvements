"""SDXL trainer wrapper for backwards compatibility."""
from typing import Optional, List, Union
import torch
from src.core.logging import get_logger, WandbLogger
from src.data.config import Config
from src.models.sdxl import StableDiffusionXLModel
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.training.trainers.base import BaseTrainer

logger = get_logger(__name__)

def create_trainer(
    config: Config,
    model: StableDiffusionXLModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    training_method: BaseTrainer,
    wandb_logger: Optional[WandbLogger] = None,
    validation_prompts: Optional[List[str]] = None
) -> SDXLTrainer:
    """Wrapper around SDXLTrainer.create for backward compatibility."""
    logger.debug("Creating trainer with configuration:")
    logger.debug(f"Device: {device}")
    logger.debug(f"Model type: {type(model).__name__}")
    logger.debug(f"Optimizer type: {type(optimizer).__name__}")

    # Ensure device is properly set
    if isinstance(device, str):
        device = torch.device(device)

    try:
        # Create trainer with explicit device
        trainer = SDXLTrainer.create(
            config=config,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=device,
            training_method=training_method,
            wandb_logger=wandb_logger,
            validation_prompts=validation_prompts
        )

        logger.info("Trainer created successfully")
        return trainer

    except Exception as e:
        logger.error(
            "Failed to create trainer",
            exc_info=True,
            extra={
                'config': str(config),
                'device': str(device),
                'error': str(e)
            }
        )
        raise
