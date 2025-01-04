"""Main orchestration script for SDXL fine-tuning """
import os

def get_unique_port():
    """Get a unique port in a range unlikely to be used by other programs."""
    # Use range 50000-55000 which is typically unused
    base_port = 50000
    port_range = 5000
    # Use RANK to offset port and avoid conflicts between different runs
    rank_offset = int(os.environ.get("RANK", "0")) * 10
    return base_port + rank_offset

# Set environment variables before any other imports
def setup_training_env():
    """Setup training environment variables."""
    # Ensure these are set before ANY imports or initialization
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(get_unique_port()))
    
    # Disable accelerate's automatic initialization
    os.environ["ACCELERATE_DISABLE_RICH"] = "1"
    os.environ["ACCELERATE_USE_RICH"] = "0"
    
    # Force PyTorch to use spawn method
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Call setup before anything else
setup_training_env()

# Now import other modules
import sys
from contextlib import contextmanager
from pathlib import Path
import torch
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
import socket
import random

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

logger, tensor_logger = setup_logging(
    log_dir="outputs/logs",
    filename="training.log",
    console_level="INFO",
    capture_warnings=True
)

CONFIG_PATH = Path("src/config.yaml")

def find_free_port():
    """Find a free port to use for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@contextmanager
def setup_environment():
    """Setup distributed training environment."""
    try:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if not torch.distributed.is_initialized():
                # Try multiple ports if the default one is in use
                max_tries = 5
                for _ in range(max_tries):
                    try:
                        port = find_free_port()
                        os.environ["MASTER_PORT"] = str(port)
                        init_process_group(backend="nccl")
                        setup_distributed()
                        logger.info("Successfully initialized distributed training")
                        break
                    except Exception as e:
                        if _ == max_tries - 1:
                            raise RuntimeError(f"Failed to initialize distributed training after {max_tries} attempts") from e
                        logger.warning(f"Port {port} failed, trying another...")
                        continue
        yield
    finally:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if torch.distributed.is_initialized():
                cleanup_distributed()
        torch_sync()

def main():
    """Main training entry point."""
    logger.info("Starting training script...", extra={'keyword': 'start'})
    
    try:
        config = Config.from_yaml(CONFIG_PATH)
        
        with setup_environment():
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}", extra={'success': True})
            
            # Initialize model and dataset in one step
            model = StableDiffusionXL.from_pretrained(
                config.model.pretrained_model_name,
                device=device,
                model_type=ModelType[config.model.model_type.upper()]
            )
            
            # Use create_dataset function which handles both dataset creation and model initialization
            dataset = create_dataset(config=config, model=model)
            train_dataloader = DataLoader(
                dataset, 
                **config.training.dataloader_kwargs,
                multiprocessing_context='spawn'  # Force spawn for all workers
            )
            
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
