"""Main training script for SDXL fine-tuning."""
import argparse
import logging
from pathlib import Path
from typing import Dict

# Third-party imports
import torch
from torch.distributed import init_process_group

# Local imports - core functionality
from src.core.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.core.logging import setup_logging, WandbLogger
from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_gc
)
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)

# Local imports - data handling
from src.data.config import Config
from src.data.dataset import create_dataset
from src.data.preprocessing import LatentPreprocessor

# Local imports - model components
from src.models.sdxl import StableDiffusionXLModel, StableDiffusionXLPipeline
from src.models.base import ModelType

# Local imports - training components
from src.training.trainer import create_trainer

# Local imports - utilities
from src.utils.paths import convert_path_list

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SDXL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def load_models(config: Config) -> Dict[str, torch.nn.Module]:
    """Load and configure all required models."""
    # Initialize SDXL model
    sdxl_model = StableDiffusionXLModel(ModelType.BASE)
    
    # Load pipeline components into model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.model.pretrained_model_name,
        torch_dtype=torch.float32
    )
    
    # Transfer pipeline components to our model
    sdxl_model.unet = pipeline.unet
    sdxl_model.vae = pipeline.vae
    sdxl_model.text_encoder_1 = pipeline.text_encoder
    sdxl_model.text_encoder_2 = pipeline.text_encoder_2
    sdxl_model.tokenizer_1 = pipeline.tokenizer
    sdxl_model.tokenizer_2 = pipeline.tokenizer_2
    sdxl_model.noise_scheduler = pipeline.scheduler
    
    # Clean up pipeline
    del pipeline
    torch_gc()

    return {
        "tokenizer_one": sdxl_model.tokenizer_1,
        "tokenizer_two": sdxl_model.tokenizer_2,
        "text_encoder_one": sdxl_model.text_encoder_1,
        "text_encoder_two": sdxl_model.text_encoder_2,
        "vae": sdxl_model.vae,
        "model": sdxl_model
    }

def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Setup distributed training if needed
    if args.local_rank != -1:
        init_process_group(backend="nccl")
        setup_distributed()
    
    # Setup logging
    setup_logging(
        log_dir=config.global_config.output_dir,
        filename="train.log"
    )
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process():
        logger.info(f"Using device: {device}")
        
    # Load and configure models
    logger.info("Loading models...")
    models = load_models(config)
    sdxl_model = models["model"]
    
    # Move models to device efficiently
    for name, model in models.items():
        if hasattr(model, 'state_dict') and not tensors_match_device(model.state_dict(), device):
            logger.info(f"Moving {name} to device {device}")
            with create_stream_context(torch.cuda.current_stream()):
                tensors_to_device_(model.state_dict(), device)
            torch_gc()
    
    # Setup memory optimizations
    setup_memory_optimizations(
        sdxl_model.unet,
        config,
        device
    )
    
    if is_main_process():
        verify_memory_optimizations(sdxl_model.unet, config, device, logger)
    
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
    # Handle multiple training directories
    image_paths = []
    captions = []
    train_dirs = config.data.train_data_dir if isinstance(config.data.train_data_dir, list) else [config.data.train_data_dir]
    
    from utils.paths import convert_path_list
    train_dirs = convert_path_list(train_dirs)
    
    for data_dir in train_dirs:
        dir_path = Path(data_dir)
        logger.info(f"Processing dataset directory: {dir_path}")
        if not dir_path.exists():
            logger.warning(f"Training directory not found: {data_dir}")
            continue
            
        # Collect images and captions from this directory
        dir_images = list(dir_path.glob("*.jpg"))
        if not dir_images:
            logger.warning(f"No jpg images found in {data_dir}")
            continue
            
        # Check for corresponding caption files
        valid_images = []
        valid_captions = []
        for img_path in dir_images:
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                try:
                    caption = caption_path.read_text(encoding='utf-8').strip()
                    valid_images.append(str(img_path))
                    valid_captions.append(caption)
                except Exception as e:
                    logger.warning(f"Error reading caption file {caption_path}: {e}")
            else:
                logger.warning(f"Missing caption file for {img_path}")
        
        image_paths.extend(valid_images)
        captions.extend(valid_captions)
        
        logger.info(f"Found {len(valid_images)} valid image-caption pairs in {data_dir}")
    
    if not image_paths:
        raise ValueError("No valid training image-caption pairs found in specified directories")
        
    logger.info(f"Total training samples: {len(image_paths)} across {len(train_dirs)} directories")
    
    train_dataset = create_dataset(
        config,
        image_paths,
        captions,
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
    trainer = create_trainer(
        config=config,
        model=sdxl_model,
        optimizer=optimizer,
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
