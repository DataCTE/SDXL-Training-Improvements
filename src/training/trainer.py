"""SDXL trainer wrapper for backward compatibility."""
from typing import List, Optional, Union

import torch
from src.core.logging import setup_logging, WandbLogger
from src.data.config import Config
from src.models.sdxl import StableDiffusionXLModel, StableDiffusionXLPipeline
from src.training.trainers.sdxl_trainer import SDXLTrainer

logger = setup_logging(__name__)

def create_trainer(
    config: Config,
    model: StableDiffusionXLModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
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
    
    # Create trainer with explicit device
    trainer = SDXLTrainer.create(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        device=device,
        wandb_logger=wandb_logger,
        validation_prompts=validation_prompts
    )
    
    # Verify data format compatibility
    sample_batch = next(iter(train_dataloader))
    if not any(k in sample_batch for k in ["latent", "model_input"]):
        raise ValueError("DataLoader must provide either 'latent' or 'model_input'")
    if not any(k in sample_batch for k in ["embeddings", "prompt_embeds"]):
        raise ValueError("DataLoader must provide either 'embeddings' or 'prompt_embeds'")
        
    return trainer
