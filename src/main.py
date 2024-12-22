"""Main orchestration script for SDXL fine-tuning with enhanced error handling and resource management."""
import argparse
import logging
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.distributed import init_process_group

from src.core.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.core.logging import setup_logging, WandbLogger
from src.data.preprocessing import CacheManager
from src.data.utils.paths import convert_windows_path
from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_sync
)
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)
from src.data.config import Config
from src.data.dataset import create_dataset
from src.data.preprocessing import LatentPreprocessor
from src.training.trainer import create_trainer
from src.data.utils.paths import convert_path_list
from src.data.dataset import create_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from src.models import ModelType, StableDiffusionXLModel

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
        torch_sync()

def setup_device_and_logging(config: Config) -> torch.device:
    """Initialize device and setup logging."""
    # Specify CUDA device index explicitly
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
            logger.info(f"cuda Device: {torch.cuda.get_device_name(device.index)}")
            logger.info(f"cuda Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.1f} GB")
            
    return device

def setup_model(config: Config, device: torch.device) -> StableDiffusionXLModel:
    """Initialize and load SDXL model.
    
    Args:
        config: Training configuration
        device: Target device
        
    Returns:
        Initialized SDXL model
    """
    logger.info("Loading models...")
    try:
        # Create base model
        model = StableDiffusionXLModel(ModelType.BASE)
        
        # Load VAE
        logger.info("Loading VAE...")
        model.vae = AutoencoderKL.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Load text encoders
        logger.info("Loading text encoders...")
        model.text_encoder_1 = CLIPTextModel.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True
        )
        model.text_encoder_2 = CLIPTextModel.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Load UNet
        logger.info("Loading UNet...")
        model.unet = UNet2DConditionModel.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Load tokenizers
        logger.info("Loading tokenizers...")
        model.tokenizer_1 = CLIPTokenizer.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="tokenizer"
        )
        model.tokenizer_2 = CLIPTokenizer.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="tokenizer_2"
        )
        
        # Move model to device
        logger.info(f"Moving model to {device}")
        model.to(device)
        
        return model
        
    except Exception as e:
        error_context = {
            'model_name': config.model.pretrained_model_name,
            'device': str(device),
            'error': str(e)
        }
        logger.error("Failed to initialize model", extra=error_context)
        raise RuntimeError("Failed to load models") from e

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
    model: StableDiffusionXLModel,
    device: torch.device,
    image_paths: List[str],
    captions: List[str]
) -> tuple[torch.utils.data.DataLoader, torch.optim.Optimizer, Optional[WandbLogger]]:
    """Setup training components."""
    try:
        # Initialize latent preprocessor
        latent_preprocessor = LatentPreprocessor(
            config=config,
            sdxl_model=model,
            device=device
        )
        
        
        if config.global_config.cache.clear_cache_on_start:
            logger.info("Clearing latent cache...")
            latent_preprocessor.clear_cache()
            
        
        # Initialize cache manager
        cache_manager = CacheManager(
            cache_dir=Path(convert_windows_path(config.global_config.cache.cache_dir)),
            num_proc=config.global_config.cache.num_proc,
            chunk_size=config.global_config.cache.chunk_size,
            compression=getattr(config.global_config.cache, 'compression', 'zstd'),
            verify_hashes=config.global_config.cache.verify_hashes,
            max_memory_usage=0.8,
            enable_memory_tracking=True
        )

        # Create and preprocess dataset
        train_dataset = create_dataset(
            config=config,
            image_paths=image_paths,
            captions=captions,
            latent_preprocessor=latent_preprocessor,
            enable_memory_tracking=True,
            max_memory_usage=0.8,  # Set memory limit
            cache_manager=cache_manager
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
            model.unet.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.optimizer_betas,
            weight_decay=config.training.weight_decay,
            eps=config.training.optimizer_eps
        )
        
        # Setup distributed model if needed
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            model.unet = torch.nn.parallel.DistributedDataParallel(
                model.unet,
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
            wandb_logger.log_model(model.unet)
            
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
            models = setup_model(config, device)
            
            # Move model to device with enhanced memory management
            if hasattr(models, 'state_dict') and not tensors_match_device(models.state_dict(), device):
                logger.info(f"Moving model to device {device}")
                try:
                    # Create dedicated stream for transfers
                    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                    
                    # Track memory before transfer
                    pre_transfer_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    with create_stream_context(stream):
                        try:
                            tensors_to_device_(models.state_dict(), device)
                        finally:
                            # Ensure proper stream synchronization
                            if stream:
                                stream.synchronize()
                            
                            # Clean up after transfer
                            torch_sync()
                    
                    # Track memory impact
                    if torch.cuda.is_available():
                        post_transfer_memory = torch.cuda.memory_allocated()
                        memory_delta = post_transfer_memory - pre_transfer_memory
                        logger.debug(f"Memory delta: {memory_delta / 1024**2:.1f}MB")
                
                except Exception as e:
                    raise TrainingSetupError(
                        "Failed to move models to device",
                        {
                            "error": str(e),
                            "device": str(device),
                            "cuda_available": torch.cuda.is_available(),
                            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        }
                    )
            
            # Setup memory optimizations
            setup_memory_optimizations(models.unet, config, device)
            if is_main_process():
                verify_memory_optimizations(models.unet, config, device, logger)
            
            # Load training data
            logger.info("Loading training data...")
            image_paths, captions = load_training_data(config)
            
            # Setup training components
            logger.info("Setting up training...")
            train_dataloader, optimizer, wandb_logger = setup_training(
                config=config,
                model=models,
                device=device,
                image_paths=image_paths,
                captions=captions
            )
            
            # Create and execute trainer
            trainer = create_trainer(
                config=config,
                model=models,
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

