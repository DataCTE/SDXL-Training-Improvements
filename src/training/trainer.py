"""SDXL trainer factory and base implementation."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Type

import torch
from tqdm.auto import tqdm
from diffusers import DDPMScheduler

from src.core.types import DataType, ModelWeightDtypes
from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_gc
)
from src.data.config import Config
from src.core.distributed import is_main_process
from src.core.logging import WandbLogger, log_metrics
from src.models import StableDiffusionXLModel
from src.training.trainers.base import TrainingMethod
from src.training.methods.ddpm_trainer import DDPMTrainer 
from src.training.methods.flow_matching_trainer import FlowMatchingTrainer
from src.training.trainers.SDXLTrainer import SDXLTrainer

logger = logging.getLogger(__name__)

def create_trainer(
    config: Config,
    model: StableDiffusionXLModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    wandb_logger: Optional[WandbLogger] = None,
    validation_prompts: Optional[List[str]] = None
) -> SDXLTrainer:
    """Create appropriate trainer based on config.
    
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
    # Map method names to trainer classes
    trainer_map = {
        "ddpm": DDPMTrainer,
        "flow_matching": FlowMatchingTrainer
    }
    
    # Get trainer class
    method = config.training.method.lower()
    if method not in trainer_map:
        raise ValueError(f"Unknown training method: {method}")
    
    trainer_cls = trainer_map[method]
    logger.info(f"Using {trainer_cls.__name__} for training")
    
    # Create trainer instance
    # Create training method instance
    training_method = trainer_cls(
        unet=model.unet,
        config=config
    )
    
    # Create and return trainer
    return SDXLTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        training_method=training_method,
        device=device,
        wandb_logger=wandb_logger,
        validation_prompts=validation_prompts
    )
    
        
