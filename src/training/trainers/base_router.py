from typing import Optional, Type, Dict
import torch
from torch.utils.data import DataLoader
from importlib import import_module
from src.core.logging import WandbLogger, get_logger
from src.data.config import Config
from src.models.base import ModelType

logger = get_logger(__name__)

class BaseRouter:
    """Base router class that dynamically handles training based on model type."""
    
    # Map model types to their trainer modules
    TRAINER_MAP = {
        ModelType.SDXL: "src.training.trainers.sdxl_trainer",
        ModelType.SD: "src.training.trainers.sd_trainer",
        ModelType.IF: "src.training.trainers.if_trainer",
        # Add more model types as needed
    }
    
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
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.wandb_logger = wandb_logger
        self.config = config
        
        # Dynamically get the appropriate trainer
        self.trainer = self._get_model_trainer()

    def _get_model_trainer(self):
        """Dynamically import and initialize the appropriate trainer based on model type."""
        try:
            # Get model type from config
            model_type = ModelType[self.config.model.model_type.upper()]
            
            # Get trainer module path
            trainer_module_path = self.TRAINER_MAP.get(model_type)
            if not trainer_module_path:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Dynamically import trainer module
            module_path, class_name = trainer_module_path.rsplit('.', 1)
            trainer_module = import_module(module_path)
            trainer_class = getattr(trainer_module, f"{model_type.value}Trainer")
            
            logger.info(f"Initializing {trainer_class.__name__} for model type: {model_type}")
            
            # Initialize trainer
            return trainer_class(
                model=self.model,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                device=self.device,
                wandb_logger=self.wandb_logger,
                config=self.config
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}", exc_info=True)
            raise

    def train(self, num_epochs: int):
        """Route training to the appropriate model trainer."""
        try:
            logger.info("Starting training process...")
            
            # Validate configuration
            if not self.config:
                raise ValueError("Configuration is required for training")
                
            # Log training parameters
            logger.info(f"Training for {num_epochs} epochs")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Model type: {self.config.model.model_type}")
            logger.info(f"Training method: {self.config.training.method}")
            
            # Route to appropriate trainer
            self.trainer.train(num_epochs)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save training checkpoint."""
        from src.training.trainers import save_checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            config=self.config,
            is_final=is_final
        ) 