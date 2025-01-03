"""Main orchestration script for SDXL fine-tuning """
import os
import multiprocessing as mp
import torch.cuda
import sys
from contextlib import contextmanager
from pathlib import Path
import torch
from torch.distributed import init_process_group
from torch.utils.data import DataLoader

# Core imports
from src.core.logging import setup_logging, get_logger, WandbLogger
from src.core.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.core.memory import torch_sync

# Data and model imports
from src.data.config import Config
from src.data.dataset import create_dataset
from src.models import StableDiffusionXL
from src.models.base import ModelType
from src.training.trainers import BaseRouter

# Import our custom optimizers
from src.training.optimizers import AdamWBF16, AdamWScheduleFreeKahan, SOAP

# Setup enhanced logging first
logger, tensor_logger = setup_logging(
    log_dir="outputs/logs",
    filename="training.log",
    console_level="INFO",
    capture_warnings=True
)

CONFIG_PATH = Path("src/config.yaml")

@contextmanager
def setup_environment():
    """Setup distributed training environment."""
    try:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            init_process_group(backend="nccl")
            setup_distributed()
        yield
    finally:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            cleanup_distributed()
        torch_sync()

def main():
    """Main training entry point."""
    logger.info("Starting training script...", extra={'keyword': 'start'})
    mp.set_start_method('spawn', force=True)
    
    # Set default distributed training environment variables
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    
    try:
        config = Config.from_yaml(CONFIG_PATH)
        
        with setup_environment():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}", extra={'success': True})
            
            # Initialize model
            model = StableDiffusionXL.from_pretrained(
                config.model.pretrained_model_name,
                device=device,
                model_type=ModelType[config.model.model_type.upper()]
            )
            
            # Create dataset and dataloader
            dataset = create_dataset(config=config, model=model)
            train_dataloader = DataLoader(dataset, **config.training.dataloader_kwargs)
            logger.info(f"Created dataloader with {len(train_dataloader)} batches")
            
            # Get optimizer class from our custom implementations
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
            
            # Initialize wandb logger
            wandb_logger = WandbLogger(config) if config.global_config.logging.use_wandb else None
            
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
            
            # Save final model if configured
            if config.training.save_final_model:
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
