from typing import Optional, Type, Dict
import torch
from torch.utils.data import DataLoader
from importlib import import_module
from src.core.logging import WandbLogger, get_logger
from src.data.config import Config
from src.models.base import ModelType
from abc import ABC, abstractmethod

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
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
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.wandb_logger = wandb_logger
        self.config = config
        
    @abstractmethod
    def train(self, num_epochs: int) -> None:
        """Train the model."""
        pass

class BaseRouter:
    """Router class that dynamically handles training based on model type."""
    
    # Map model types to their trainer modules
    TRAINER_MAP = {
        ModelType.SDXL: "src.training.trainers.sdxl_trainer.SDXLTrainer",
        #ModelType.SD: "src.training.trainers.sd_trainer.SDTrainer",
        #ModelType.IF: "src.training.trainers.if_trainer.IFTrainer",
    }
    
    @classmethod
    def create(
        cls,
        model,
        optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
        wandb_logger: Optional[WandbLogger] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """Factory method to create the appropriate trainer."""
        try:
            model_type = ModelType[config.model.model_type.upper()]
            trainer_path = cls.TRAINER_MAP.get(model_type)
            if not trainer_path:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            module_path, class_name = trainer_path.rsplit('.', 1)
            trainer_module = import_module(module_path)
            trainer_class = getattr(trainer_module, class_name)
            
            logger.info(f"Creating {trainer_class.__name__} for model type: {model_type}")
            
            return trainer_class(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {str(e)}", exc_info=True)
            raise 