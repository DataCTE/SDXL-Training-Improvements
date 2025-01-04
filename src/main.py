"""Main orchestration script for SDXL fine-tuning."""
import os
from src.core.distributed import (
    setup_training_env,
    setup_environment,
    convert_model_to_ddp,
    is_main_process,
    get_world_size
)

# Must be called before ANY other imports
setup_training_env()

# Now safe to import the rest
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Core imports
from src.core.logging import setup_logging, get_logger, WandbLogger
from src.data.config import Config
from src.data.dataset import create_dataset
from src.models import StableDiffusionXL
from src.models.base import ModelType
from src.training.trainers import BaseRouter
from src.training.optimizers import AdamWBF16, AdamWScheduleFreeKahan, SOAP

logger, tensor_logger = setup_logging(
    log_dir="outputs/logs",
    filename="training.log",
    console_level="INFO",
    capture_warnings=True
)

CONFIG_PATH = Path("src/config.yaml")

def main():
    """Main training entry point."""
    logger.info("Starting training script...", extra={'keyword': 'start'})
    
    try:
        config = Config.from_yaml(CONFIG_PATH)
        
        with setup_environment():
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}", extra={'success': True})
            
            # Initialize model
            model = StableDiffusionXL.from_pretrained(
                config.model.pretrained_model_name,
                device=device,
                model_type=ModelType[config.model.model_type.upper()]
            )
            
            # Convert to DDP if using distributed training
            if get_world_size() > 1:
                model = convert_model_to_ddp(
                    model,
                    device_ids=[device.index] if device.type == "cuda" else None
                )
            
            # Create dataset and dataloader
            dataset = create_dataset(config=config, model=model)
            train_dataloader = DataLoader(
                dataset, 
                **config.training.dataloader_kwargs,
                multiprocessing_context='spawn'  # Force spawn for all workers
            )
            
            # Setup optimizer
            optimizer_map = {
                "AdamWBF16": AdamWBF16,
                "AdamWScheduleFreeKahan": AdamWScheduleFreeKahan,
                "SOAP": SOAP
            }
            
            optimizer_class = optimizer_map.get(config.optimizer.class_name)
            if optimizer_class is None:
                raise ValueError(f"Unsupported optimizer: {config.optimizer.class_name}")
            
            optimizer = optimizer_class(
                model.parameters(),
                **config.optimizer.kwargs
            )
            
            # Initialize wandb logger only on main process
            wandb_logger = None
            if is_main_process() and config.global_config.logging.use_wandb:
                wandb_logger = WandbLogger(config)
            
            # Create trainer
            trainer = BaseRouter.create(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config
            )
            
            # Start training loop
            logger.info("Starting training loop...")
            trainer.train(num_epochs=config.training.num_epochs)
            
            # Save final model if configured (only on main process)
            if config.training.save_final_model and is_main_process():
                save_path = Path("outputs/models/final_model")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(save_path, is_final=True)
                logger.info(f"Saved final model to {save_path}")
            
            logger.info("Training completed successfully", extra={'keyword': 'success'})

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
