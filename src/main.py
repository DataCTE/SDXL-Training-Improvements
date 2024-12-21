"""Main orchestration script for SDXL fine-tuning with enhanced error handling and resource management."""
import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.distributed import init_process_group

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
from src.data.config import Config
from src.data.dataset import create_dataset
from src.data.preprocessing import LatentPreprocessor
from src.models.sdxl import StableDiffusionXLModel, StableDiffusionXLPipeline
from src.models.base import ModelType
from src.training.trainer import create_trainer
from src.data.utils.paths import convert_path_list
from src.data.dataset import create_dataset
logger = logging.getLogger(__name__)

class TrainingSetupError(Exception):
    """Base exception for training setup errors."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message)
        self.context = context or {}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with validation."""
    parser = argparse.ArgumentParser(description="SDXL Fine-tuning Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Validate config path
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    return args

@contextmanager
def setup_environment(args: argparse.Namespace):
    """Setup training environment with proper cleanup."""
    try:
        # Setup distributed training if needed
        if args.local_rank != -1:
            init_process_group(backend="nccl")
            setup_distributed()
            
        yield
        
    finally:
        cleanup_distributed()
        torch_gc()

def setup_device_and_logging(config: Config) -> torch.device:
    """Initialize device and setup logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging directories
    output_dir = Path(config.global_config.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir=str(log_dir), filename="train.log")
    
    # Log device info if main process
    if is_main_process():
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
            
    return device

def load_models(config: Config, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load and configure models with error handling."""
    try:
        # Initialize base model
        sdxl_model = StableDiffusionXLModel(ModelType.BASE)
        
        # Configure model loading
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        device_map = "balanced" if device.type == "cuda" else None
        
        # Load pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            config.model.pretrained_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        
        # Transfer components
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
            "unet": sdxl_model.unet,
            "model": sdxl_model
        }
        
    except Exception as e:
        raise TrainingSetupError(
            "Failed to load models",
            {"error": str(e), "device": str(device)}
        )

def load_training_data(config: Config) -> tuple[List[str], List[str]]:
    """Load and validate training data."""
    image_paths = []
    captions = []
    
    # Process training directories
    train_dirs = (config.data.train_data_dir if isinstance(config.data.train_data_dir, list) 
                 else [config.data.train_data_dir])
    train_dirs = convert_path_list(train_dirs)
    
    for data_dir in train_dirs:
        dir_path = Path(data_dir)
        logger.info(f"Processing dataset directory: {dir_path}")
        
        if not dir_path.exists():
            logger.warning(f"Training directory not found: {data_dir}")
            continue
        
        # Collect images
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        dir_images = []
        for ext in image_extensions:
            dir_images.extend(list(dir_path.glob(ext)))
            
        if not dir_images:
            logger.warning(
                f"No images found in {data_dir} "
                f"(supported: {', '.join(image_extensions)})"
            )
            continue
            
        # Process image-caption pairs
        valid_pairs = process_image_caption_pairs(dir_images)
        image_paths.extend(valid_pairs[0])
        captions.extend(valid_pairs[1])
        
        logger.info(f"Found {len(valid_pairs[0])} valid pairs in {data_dir}")
        
    if not image_paths:
        raise TrainingSetupError("No valid training data found")
        
    logger.info(f"Total training samples: {len(image_paths)} across {len(train_dirs)} directories")
    return image_paths, captions

def process_image_caption_pairs(image_files: List[Path]) -> tuple[List[str], List[str]]:
    """Process and validate image-caption pairs."""
    valid_images = []
    valid_captions = []
    
    for img_path in image_files:
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            try:
                caption = caption_path.read_text(encoding='utf-8').strip()
                if caption:  # Skip empty captions
                    valid_images.append(str(img_path))
                    valid_captions.append(caption)
            except Exception as e:
                logger.warning(f"Error reading caption file {caption_path}: {e}")
        else:
            logger.warning(f"Missing caption file for {img_path}")
            
    return valid_images, valid_captions

def setup_training(
    config: Config,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    image_paths: List[str],
    captions: List[str]
) -> tuple[torch.utils.data.DataLoader, torch.optim.Optimizer, Optional[WandbLogger]]:
    """Setup training components."""
    try:
        # Initialize latent preprocessor
        latent_preprocessor = LatentPreprocessor(
            config,
            models["tokenizer_one"],
            models["tokenizer_two"],
            models["text_encoder_one"],
            models["text_encoder_two"],
            models["vae"],
            device,
            use_cache=config.global_config.cache.use_cache
        )
        
        if config.global_config.cache.clear_cache_on_start:
            logger.info("Clearing latent cache...")
            latent_preprocessor.clear_cache()
        
        # Create and preprocess dataset
        train_dataset = create_dataset(
            config=config,
            image_paths=image_paths,
            captions=captions,
            latent_preprocessor=latent_preprocessor,
            enable_memory_tracking=True,
            max_memory_usage=0.8  # Set memory limit
        )
        
        if config.global_config.cache.use_cache:
            train_dataset = latent_preprocessor.preprocess_dataset(
                train_dataset,
                batch_size=config.training.batch_size,
                cache=True,
                compression=getattr(config.global_config.cache, 'compression', 'zstd')
            )
        
        # Create data loader
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
        
        # Setup distributed model if needed
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            models["unet"] = torch.nn.parallel.DistributedDataParallel(
                models["unet"],
                device_ids=[device] if device.type == "cuda" else None
            )
        
        # Initialize wandb logger
        wandb_logger = None
        if config.training.use_wandb and is_main_process():
            wandb_logger = WandbLogger(
                project="sdxl-training",
                name=Path(config.global_config.output_dir).name,
                config=config.__dict__,
                dir=config.global_config.output_dir,
                tags=["sdxl", "fine-tuning"]
            )
            wandb_logger.log_model(models["unet"])
            
        return train_dataloader, optimizer, wandb_logger
        
    except Exception as e:
        raise TrainingSetupError(
            "Failed to setup training components",
            {"error": str(e)}
        )

def main():
    """Main orchestration function with enhanced error handling."""
    try:
        args = parse_args()
        
        # Load configuration
        config = Config.from_yaml(args.config)
        
        with setup_environment(args):
            # Setup device and logging
            device = setup_device_and_logging(config)
            
            # Load models
            logger.info("Loading models...")
            models = load_models(config, device)
            
            # Move models to device
            for name, model in models.items():
                if hasattr(model, 'state_dict') and not tensors_match_device(model.state_dict(), device):
                    logger.info(f"Moving {name} to device {device}")
                    with create_stream_context(torch.cuda.current_stream()):
                        tensors_to_device_(model.state_dict(), device)
                        torch_gc()
            
            # Setup memory optimizations
            setup_memory_optimizations(models["model"].unet, config, device)
            if is_main_process():
                verify_memory_optimizations(models["model"].unet, config, device, logger)
            
            # Load training data
            logger.info("Loading training data...")
            image_paths, captions = load_training_data(config)
            
            # Setup training components
            logger.info("Setting up training...")
            train_dataloader, optimizer, wandb_logger = setup_training(
                config, models, device, image_paths, captions
            )
            
            # Create and execute trainer
            trainer = create_trainer(
                config=config,
                model=models["model"],
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger
            )
            
            logger.info("Starting training...")
            trainer.train()
            
            if is_main_process():
                logger.info("Training complete!")
                
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if isinstance(e, TrainingSetupError) and e.context:
            logger.error(f"Error context: {e.context}")
        sys.exit(1)

if __name__ == "__main__":
    main()