"""Main orchestration script for SDXL fine-tuning with 100x speedups."""
import os
import time
import multiprocessing as mp
import torch.cuda
import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
import wandb

# Core imports
from src.core.logging import setup_logging, get_logger, WandbLogger
from src.core.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.core.memory import torch_sync

# Data and model imports
from src.data.config import Config
from src.data.dataset import create_dataset
from src.models import StableDiffusionXL
from src.training.trainers import DDPMTrainer, FlowMatchingTrainer, save_checkpoint
from src.training.optimizers import (
    AdamWBF16,
    AdamWScheduleFreeKahan,
    SOAP
)
from src.data.utils.paths import load_data_from_directory

# Disable TorchDynamo in DataLoader workers
os.environ['TORCHDYNAMO_DISABLE'] = '1'

logger = get_logger(__name__)
CONFIG_PATH = Path("src/config.yaml")

@contextmanager
def setup_environment():
    """Setup distributed training environment."""
    try:
        if torch.cuda.is_available():
            init_process_group(backend="nccl")
            setup_distributed()
        yield
    finally:
        cleanup_distributed()
        torch_sync()

def setup_device_and_logging(config: Config) -> torch.device:
    """Setup device and logging based on config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process():
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            logger.info(f"Mixed precision: {config.training.mixed_precision}")
            if config.training.enable_xformers:
                logger.info("xFormers optimization enabled")

    return device

def setup_model(config: Config, device: torch.device) -> StableDiffusionXL:
    """Initialize model using config."""
    try:
        logger.info(f"Initializing {config.model.model_type} model")
        return StableDiffusionXL.from_pretrained(
            config.model.pretrained_model_name,
            device=device,
            model_type=config.model.model_type
        )
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def main():
    """Main training entry point."""
    print("Starting SDXL training script...")
    mp.set_start_method('spawn', force=True)
    
    try:
        # Load config as single source of truth
        config = Config.from_yaml(CONFIG_PATH)
        logger, _ = setup_logging(config.global_config.logging)
        
        device = None
        with setup_environment():
            device = setup_device_and_logging(config)
            model = setup_model(config, device)
            
            # Create dataset using config
            dataset = create_dataset(
                config=config,
                model=model,
                verify_cache=True
            )
            
            # Create data loader
            train_dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.training.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=config.training.pin_memory
            )
            
            # Initialize optimizer based on config type
            optimizer_map = {
                "adamw": torch.optim.AdamW,
                "adamw_bf16": AdamWBF16,
                "adamw_schedule_free_kahan": AdamWScheduleFreeKahan,
                "soap": SOAP
            }
            
            optimizer_class = optimizer_map.get(config.optimizer.optimizer_type.lower())
            if optimizer_class is None:
                raise ValueError(f"Unknown optimizer type: {config.optimizer.optimizer_type}")
                
            optimizer = optimizer_class(
                model.parameters(),
                lr=config.optimizer.learning_rate,
                weight_decay=config.optimizer.weight_decay,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.epsilon
            )

            # Initialize wandb logger
            wandb_logger = WandbLogger(config) if config.global_config.logging.use_wandb else None

            # Create trainer
            trainer_class = DDPMTrainer if config.training.method == "ddpm" else FlowMatchingTrainer
            trainer = trainer_class(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config
            )
            
            # Train
            logger.info(f"Starting training with {config.training.method} for {config.training.num_epochs} epochs...")
            trainer.train(num_epochs=config.training.num_epochs)
            
            # Save final model if configured
            if is_main_process() and config.training.save_final_model:
                logger.info("Saving final model...")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=config.training.num_epochs,
                    config=config,
                    is_final=True
                )

    except Exception as e:
        error_context = {
            'error_type': type(e).__name__,
            'device': str(device) if device is not None else 'unknown',
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'error': str(e),
            'stack_trace': traceback.format_exc()
        }
        logger.error("Training failed", exc_info=True, extra=error_context)
        sys.exit(1)

if __name__ == "__main__":
    main()
