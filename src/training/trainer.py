"""SDXL trainer factory implementation."""
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

class TrainerFactory:
    """Factory class for creating SDXL trainers."""
    
    @staticmethod
    def create_trainer(
        cls,
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
            
        Raises:
            ValueError: If training method is not registered
        """
        # Get trainer class using metaclass registry
        method = config.training.method.lower()
        trainer_cls = BaseTrainingMethod.get_method(method)
        logger.info(f"Creating trainer: {trainer_cls.__name__}")
        
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

# Convenience function for backward compatibility
def create_trainer(*args, **kwargs) -> SDXLTrainer:
    """Wrapper around TrainerFactory.create_trainer."""
    return TrainerFactory.create_trainer(*args, **kwargs)
