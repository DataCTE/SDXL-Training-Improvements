"""Main training script for SDXL fine-tuning."""
import argparse
import logging
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from config import Config
from data import create_dataset, LatentPreprocessor
from models import UNetWrapper
from utils.distributed import setup_distributed, cleanup_distributed, is_main_process
from utils.logging import setup_logging
from utils.memory import setup_memory_optimizations, verify_memory_optimizations
from training import configure_noise_scheduler

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

    # Load UNet
    unet = UNetWrapper(
        config.model.pretrained_model_name,
        device=device,
        dtype=torch.float32
    )

    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "vae": vae,
        "unet": unet
    }

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = Config()  # TODO: Load from file
    
    # Setup logging
    setup_logging(
        log_dir=config.global_config.output_dir,
        filename="train.log"
    )
    
    # Initialize distributed training
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        logger.info(f"Using device: {device}")
        
    # Load models
    logger.info("Loading models...")
    models = load_models(config, device)
    
    # Setup memory optimizations
    setup_memory_optimizations(
        models["unet"],
        config,
        device,
        config.training.batch_size,
        config.training.batch_size // config.training.gradient_accumulation_steps
    )
    
    if accelerator.is_local_main_process:
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
        ["path/to/image1.jpg"],  # TODO: Load actual image paths
        ["caption 1"],  # TODO: Load actual captions
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
    
    # Prepare for distributed training
    unet, optimizer, train_dataloader, scheduler = accelerator.prepare(
        models["unet"],
        optimizer,
        train_dataloader,
        noise_scheduler_config["scheduler"]
    )
    
    # TODO: Implement training loop
    
    # Cleanup
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        logger.info("Training complete!")
    
if __name__ == "__main__":
    main()
