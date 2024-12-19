"""Main training script for SDXL fine-tuning.

Note: utils/ directory has been removed and its contents reorganized into core/ and other modules.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

from config import Config
from data import create_dataset, LatentPreprocessor
from models import StableDiffusionXLModel, ModelType
from core.distributed import setup_distributed, cleanup_distributed, is_main_process
from core.logging import setup_logging
from core.logging.wandb import WandbLogger
from core.memory import setup_memory_optimizations, verify_memory_optimizations
from training import configure_noise_scheduler, SDXLTrainer

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SDXL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def load_models(config: Config, device: torch.device):
    """Load and configure all required models."""
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        config.model.pretrained_model_name,
        subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        config.model.pretrained_model_name,
        subfolder="tokenizer_2"
    )

    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        config.model.pretrained_model_name,
        subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModel.from_pretrained(
        config.model.pretrained_model_name,
        subfolder="text_encoder_2"
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config.model.pretrained_model_name,
        subfolder="vae"
    )

    # Load SDXL model
    sdxl_model = StableDiffusionXLModel(ModelType.BASE)
    sdxl_model.from_pretrained(
        config.model.pretrained_model_name,
        torch_dtype=torch.float32
    )

    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "vae": vae,
        "model": sdxl_model
    }

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Setup logging
    setup_logging(
        log_dir=config.global_config.output_dir,
        filename="train.log"
    )
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process():
        logger.info(f"Using device: {device}")
        
    # Load models
    logger.info("Loading models...")
    models = load_models(config, device)
    
    # Setup memory optimizations
    setup_memory_optimizations(
        models["unet"],
        config,
        device
    )
    
    if is_main_process():
        verify_memory_optimizations(models["unet"], config, device, logger)
    
    # Initialize latent preprocessor
    latent_preprocessor = LatentPreprocessor(
        config,
        models["tokenizer_one"],
        models["tokenizer_two"],
        models["text_encoder_one"],
        models["text_encoder_two"],
        models["vae"],
        device
    )
    
    # Load datasets
    logger.info("Creating datasets...")
    train_dataset = create_dataset(
        config,
        [str(p) for p in Path(config.data.train_data_dir).glob("*.jpg")],
        [p.with_suffix(".txt").read_text().strip() 
         for p in Path(config.data.train_data_dir).glob("*.jpg")],
        latent_preprocessor=latent_preprocessor,
        is_train=True
    )
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size // config.training.gradient_accumulation_steps,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        collate_fn=train_dataset.collate_fn
    )
    
    # Configure noise scheduler
    noise_scheduler_config = configure_noise_scheduler(config, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        models["unet"].parameters(),
        lr=config.training.learning_rate,
        betas=config.training.optimizer_betas,
        weight_decay=config.training.weight_decay,
        eps=config.training.optimizer_eps
    )
    
    # Initialize distributed training if needed
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        models["unet"] = torch.nn.parallel.DistributedDataParallel(
            models["unet"],
            device_ids=[device] if device.type == "cuda" else None
        )

    # Initialize W&B logger
    wandb_logger = None
    if config.training.use_wandb and is_main_process():
        wandb_logger = WandbLogger(
            project="sdxl-training",
            name=Path(config.global_config.output_dir).name,
            config=config.__dict__,
            dir=config.global_config.output_dir,
            tags=["sdxl", "fine-tuning"]
        )
        # Log initial model architecture
        wandb_logger.log_model(models["unet"])
    
    # Create trainer
    trainer = SDXLTrainer(
        config=config,
        unet=models["unet"],
        optimizer=optimizer,
        scheduler=noise_scheduler_config["scheduler"],
        train_dataloader=train_dataloader,
        device=device,
        wandb_logger=wandb_logger
    )
    
    # Execute training
    trainer.train()
    
    # Cleanup
    cleanup_distributed()
    if is_main_process():
        logger.info("Training complete!")
    
if __name__ == "__main__":
    main()
