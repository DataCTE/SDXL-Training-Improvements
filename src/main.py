"""Main orchestration script for SDXL fine-tuning with 100x speedups."""
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
from src.training.trainers import BaseRouter, save_checkpoint

logger = get_logger(__name__)
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
    print("Starting training script...")
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
        logger, _ = setup_logging(config.global_config.logging)
        
        with setup_environment():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize components using config
            model = StableDiffusionXL.from_pretrained(
                config.model.pretrained_model_name,
                device=device,
                model_type=ModelType[config.model.model_type.upper()]
            )
            
            dataset = create_dataset(config=config, model=model)
            train_dataloader = DataLoader(dataset, **config.training.dataloader_kwargs)
            
            optimizer_class = getattr(torch.optim, config.optimizer.class_name)
            optimizer = optimizer_class(model.parameters(), **config.optimizer.kwargs)
            
            wandb_logger = WandbLogger(config) if config.global_config.logging.use_wandb else None

            # Use BaseRouter for architecture-agnostic training
            trainer = BaseRouter(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config
            )
            
            trainer.train(num_epochs=config.training.num_epochs)
            
            if is_main_process() and config.training.save_final_model:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=config.training.num_epochs,
                    config=config,
                    is_final=True
                )

    except Exception as e:
        logger.error("Training failed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
