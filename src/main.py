"""Main orchestration script for SDXL fine-tuning with 100x speedups."""
import argparse
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.distributed import init_process_group
from torch.cuda.amp import autocast

from src.core.logging import (
    setup_logging,
    get_logger,
    WandbLogger,
    TensorLogger,
    create_enhanced_logger
)
from src.core.memory import (
    create_stream_context,
    tensors_to_device_,
    tensors_match_device,
    torch_sync
)
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)
from src.data.config import Config
from src.data.dataset import create_dataset
from src.data.preprocessing import (
    PreprocessingPipeline,
    CacheManager
)
from src.training.trainers import (
    BaseRouter,
    DDPMTrainer,
    FlowMatchingTrainer,
    SDXLTrainer,
    save_checkpoint
)
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_paths
from src.models import ModelType, StableDiffusionXL
from src.core.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process
)
from src.training.optimizers import (
    AdamWBF16,
    AdamWScheduleFreeKahan,
    SOAP
)
import multiprocessing as mp
import traceback
import sys
import logging
from contextlib import contextmanager
from pathlib import Path

logger = get_logger(__name__)

class TrainingSetupError(Exception):
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message)
        self.context = context or {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL Fine-tuning Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    return args

@contextmanager
def setup_environment(args: argparse.Namespace):
    try:
        if args.local_rank != -1:
            init_process_group(backend="nccl")
            setup_distributed()
        yield
    finally:
        cleanup_distributed()
        torch_sync()

def setup_model(config: Config, device: torch.device) -> StableDiffusionXL:
    """Initialize model with proper error handling."""
    try:
        logger.info(f"Initializing {config.model.model_type} model")
        model = StableDiffusionXL.from_pretrained(
            config.model.pretrained_model_name,
            device=device,
            model_type=ModelType(config.model.model_type)
        )
        
        # Create tensor logger
        tensor_logger = TensorLogger(logger, {
            "vae": model.vae,
            "text_encoder_1": model.text_encoder_1,
            "text_encoder_2": model.text_encoder_2
        })

        # Store tensor_logger in model for access by training methods
        model.tensor_logger = tensor_logger
        
        # Verify model components
        components = {
            'VAE': model.vae,
            'Text Encoder 1': model.text_encoder_1,
            'Text Encoder 2': model.text_encoder_2,
            'Tokenizer 1': model.tokenizer_1,
            'Tokenizer 2': model.tokenizer_2,
            'UNet': model.unet,
            'Noise Scheduler': model.noise_scheduler,
            'CLIP Encoder 1': model.clip_encoder_1,
            'CLIP Encoder 2': model.clip_encoder_2,
            'VAE Encoder': model.vae_encoder
        }
        
        missing = [name for name, comp in components.items() if not comp]
        
        if missing:
            raise RuntimeError(f"Model initialization incomplete. Missing components: {', '.join(missing)}")

        logger.info("Model initialized successfully")
        return model

    except Exception as e:
        error_context = {
            'model_name': config.model.pretrained_model_name,
            'device_type': device.type,
            'device_index': device.index,
            'error': str(e),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'model_state': 'partially_loaded' if 'model' in locals() else 'not_created'
        }
        logger.error("Failed to initialize model", extra=error_context)
        raise RuntimeError(f"Failed to load models: {str(e)}") from e

def load_training_data(config: Config) -> tuple[List[str], List[str]]:
    """Load and validate training data paths."""
    image_paths, captions = [], []
    
    # Convert train directories to proper format
    train_dirs = (config.data.train_data_dir if isinstance(config.data.train_data_dir, list) 
                 else [config.data.train_data_dir])
    
    # Convert each path individually
    train_dirs = [convert_windows_path(path) for path in train_dirs]
    
    for data_dir in train_dirs:
        dir_path = Path(data_dir)
        logger.info(f"Processing dataset directory: {dir_path}")
        
        if not dir_path.exists():
            logger.warning(f"Training directory not found: {data_dir}")
            continue
            
        # Find images with supported extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        dir_images = []
        for ext in image_extensions:
            dir_images.extend(list(dir_path.glob(ext)))
            
        if not dir_images:
            logger.warning(f"No images found in {data_dir} (supported: {', '.join(image_extensions)})")
            continue
            
        # Process valid image-caption pairs
        valid_pairs = process_image_caption_pairs(dir_images)
        image_paths.extend(valid_pairs[0])
        captions.extend(valid_pairs[1])
        logger.info(f"Found {len(valid_pairs[0])} valid pairs in {data_dir}")
        
    if not image_paths:
        raise TrainingSetupError("No valid training data found")
        
    logger.info(f"Total training samples: {len(image_paths)} across {len(train_dirs)} directories")
    return image_paths, captions

def process_image_caption_pairs(image_files: List[Path]) -> tuple[List[str], List[str]]:
    valid_images, valid_captions = [], []
    for img_path in image_files:
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            try:
                caption = caption_path.read_text(encoding='utf-8').strip()
                if caption:
                    valid_images.append(str(img_path))
                    valid_captions.append(caption)
            except Exception as e:
                logger.warning(f"Error reading caption file {caption_path}: {e}")
        else:
            logger.warning(f"Missing caption file for {img_path}")
    return valid_images, valid_captions

def create_trainer(
    config: Config,
    model: StableDiffusionXL,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    wandb_logger: Optional[WandbLogger] = None,
) -> BaseRouter:
    """Create appropriate trainer based on config."""
    try:
        # Create trainer through SDXLTrainer for proper delegation
        trainer = SDXLTrainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=device,
            wandb_logger=wandb_logger,
            config=config,
            training_method=config.training.method,  # Pass method to SDXLTrainer
            # Pass additional training configuration
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            clip_grad_norm=config.training.clip_grad_norm,
            enable_xformers=config.training.enable_xformers
        )
        
        logger.info(f"Created trainer for method: {config.training.method}")
        return trainer
        
    except Exception as e:
        error_context = {
            'training_method': config.training.method,
            'model_type': config.model.model_type,
            'batch_size': config.training.batch_size,
            'error': str(e)
        }
        logger.error(
            "Failed to create trainer",
            exc_info=True,
            extra=error_context
        )
        raise TrainingSetupError("Failed to create trainer", error_context) from e

def setup_training(
    config: Config,
    model: StableDiffusionXL,
    device: torch.device,
    image_paths: List[str],
    captions: List[str]
) -> Tuple[torch.utils.data.DataLoader, torch.optim.Optimizer, Optional[WandbLogger]]:
    """Setup training components."""
    try:
        # Create cache manager
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=device
        )

        # Create preprocessing pipeline
        preprocessing_pipeline = PreprocessingPipeline(
            config=config,
            model=model,
            cache_manager=cache_manager,
            is_train=True,
            enable_memory_tracking=True
        )

        # Create dataset
        dataset = create_dataset(
            config=config,
            image_paths=image_paths,
            captions=captions,
            preprocessing_pipeline=preprocessing_pipeline,
            enable_memory_tracking=True
        )

        # Create data loader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            drop_last=True  # Drop incomplete batches
        )

        # Initialize optimizer
        optimizer_cls = {
            "adamw": torch.optim.AdamW,
            "adamw_bf16": AdamWBF16,
            "adamw_schedule_free_kahan": AdamWScheduleFreeKahan,
            "SOAP": SOAP
        }[config.optimizer.optimizer_type]
        
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.epsilon
        )

        # Initialize wandb logger if enabled
        wandb_logger = None
        if config.global_config.logging.use_wandb and is_main_process():
            # Create a descriptive run name based on training method
            method_specific_info = {
                "ddpm": f"-{config.training.prediction_type}-{config.noise_scheduler.num_train_timesteps}steps",
                "flow_matching": "-flow"
            }.get(config.training.method, "")
            
            run_name = (
                f"{config.model.model_type}"
                f"-{config.training.method}"
                f"{method_specific_info}"
                f"-{config.optimizer.optimizer_type}"
                f"-lr{config.optimizer.learning_rate:.2e}"
                f"-bs{config.training.batch_size}"
                f"-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Enhanced tags based on training method
            method_specific_tags = {
                "ddpm": [
                    f"pred_{config.training.prediction_type}",
                    f"steps_{config.noise_scheduler.num_train_timesteps}"
                ],
                "flow_matching": ["flow_matching"]
            }.get(config.training.method, [])
            
            wandb_logger = WandbLogger(
                project=config.global_config.logging.wandb_project,
                entity=config.global_config.logging.wandb_entity,
                name=run_name,
                config=config.to_dict(),
                tags=[
                    config.model.model_type,
                    config.training.method,
                    config.optimizer.optimizer_type,
                    f"bs{config.training.batch_size}",
                    config.training.mixed_precision,
                    *method_specific_tags
                ]
            )
            
            # Log wandb URL to terminal
            if wandb_logger.run is not None:
                logger.info(
                    f"\nWandB run started: {wandb_logger.run.url}\n"
                    f"View live training progress at the URL above ☝️"
                )
            
            # Log additional method-specific configuration
            method_config = {
                "training_method": config.training.method,
                "optimizer": {
                    "type": config.optimizer.optimizer_type,
                    "learning_rate": config.optimizer.learning_rate,
                    "weight_decay": config.optimizer.weight_decay,
                    "beta1": config.optimizer.beta1,
                    "beta2": config.optimizer.beta2,
                    "epsilon": config.optimizer.epsilon
                },
                "batch_size": config.training.batch_size,
                "mixed_precision": config.training.mixed_precision,
                "gradient_clipping": config.training.max_grad_norm
            }
            
            # Add method-specific config
            if config.training.method == "ddpm":
                method_config.update({
                    "prediction_type": config.training.prediction_type,
                    "num_train_timesteps": config.training.method_config.scheduler.num_train_timesteps,
                    "beta_schedule": config.training.method_config.scheduler.beta_schedule,
                    "rescale_betas_zero_snr": config.training.method_config.scheduler.rescale_betas_zero_snr
                })
            elif config.training.method == "flow_matching":
                method_config.update({
                    "flow_type": "optimal_transport"
                })
                
            wandb_logger.log_hyperparams(method_config)

        return train_dataloader, optimizer, wandb_logger

    except Exception as e:
        logger.error(
            "Failed to setup training",
            exc_info=True,
            extra={
                'config': config.to_dict(),
                'error': str(e)
            }
        )
        raise

def check_system_limits():
    """Check and attempt to increase system file limits."""
    import resource
    try:
        # Get current soft limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Try to increase soft limit to hard limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        
        logger.info(f"Increased file limit from {soft} to {hard}")
    except Exception as e:
        logger.warning(f"Could not increase file limit: {e}")
        logger.warning("You may need to increase system file limits (ulimit -n)")

def worker_init_fn(
    worker_id: int, 
    config: Config,
    device: torch.device
) -> None:
    """Initialize worker process."""
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    
    # Disable TorchDynamo in worker processes
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    
    # Set thread count for worker
    torch.set_num_threads(1)
    
    try:
        # Initialize cache manager
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=device
        )

        # Get worker's dataset instance
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Initialize model in worker
            model = setup_model(config, device)
            
            # Initialize preprocessing pipeline in the worker
            worker_info.dataset.preprocessing_pipeline.initialize_worker(
                model=model,
                cache_manager=cache_manager,
                device=device
            )
            
    except Exception as e:
        logger.error(f"Failed to initialize worker {worker_id}: {str(e)}", exc_info=True)
        raise

def setup_device(config: Config) -> torch.device:
    """Setup device with proper error handling."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            # Set device index if using distributed training
            if torch.distributed.is_initialized():
                local_rank = torch.distributed.get_rank()
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
        
        return device
        
    except Exception as e:
        logger.error(
            "Failed to setup device",
            exc_info=True,
            extra={
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'error': str(e)
            }
        )
        raise

def initialize_logging(config: Config) -> Tuple[logging.Logger, TensorLogger]:
    """Initialize logging system with proper error handling.
    
    Args:
        config: Configuration object containing logging settings
        
    Returns:
        Tuple of (main logger, tensor logger)
        
    Raises:
        RuntimeError: If logger initialization fails
    """
    try:
        # Create log directory if it doesn't exist
        log_dir = Path(config.global_config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger, tensor_logger = setup_logging(
            config=config.global_config.logging,
            log_dir=str(log_dir),
            level=config.global_config.logging.file_level,
            filename=config.global_config.logging.filename,
            module_name="sdxl_training",
            capture_warnings=config.global_config.logging.capture_warnings,
            console_level=config.global_config.logging.console_level
        )
        
        if not logger:
            raise RuntimeError("Logger initialization returned None")
            
        logger.info("Logger initialized successfully")
        logger.info(f"Loaded configuration from {config.config_path}")
        
        return logger, tensor_logger
        
    except Exception as e:
        # Use print since logger isn't available yet
        print(f"Failed to initialize logging system: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        raise RuntimeError(f"Failed to initialize logging: {str(e)}") from e

def main():
    """Main training entry point."""
    print("Starting SDXL training script...")
    mp.set_start_method('spawn', force=True)
    
    # Initialize these before try block so they're available in error handling
    device = None
    logger = None
    config = None
    
    try:
        # Parse args and load config first
        args = parse_args()
        config = Config.from_yaml(args.config)
        
        # Initialize logging system
        logger, tensor_logger = initialize_logging(config)
        
        # Check system limits
        check_system_limits()
        
        with setup_environment(args):
            # Setup device
            device = setup_device(config)
            
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                
                # Log mixed precision settings
                if config.training.mixed_precision != "no":
                    logger.info(f"Using mixed precision: {config.training.mixed_precision}")
                if config.training.enable_xformers:
                    logger.info("xFormers optimization enabled")
            else:
                logger.info("Using CPU device")
            
            model = setup_model(config, device)
            
            if model is None:
                raise RuntimeError("Failed to initialize model")
            
            logger.info("Loading training data...")
            image_paths, captions = load_training_data(config)
            
            logger.info("Setting up training...")
            train_dataloader, optimizer, wandb_logger = setup_training(
                config=config,
                model=model,
                device=device,
                image_paths=image_paths,
                captions=captions
            )
            
            # Create trainer through SDXLTrainer for proper delegation
            trainer = create_trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger
            )
            
            # Log training start
            total_steps = len(train_dataloader) * config.training.num_epochs
            logger.info(
                f"Starting training with {config.training.method} method:\n"
                f"- Total epochs: {config.training.num_epochs}\n"
                f"- Steps per epoch: {len(train_dataloader)}\n"
                f"- Total steps: {total_steps}\n"
                f"- Batch size: {config.training.batch_size}\n"
                f"- Device: {device}"
            )
            
            # Execute training
            trainer.train(num_epochs=config.training.num_epochs)
            
            if is_main_process():
                logger.info("Training complete!")
                if config.training.save_final_model:
                    logger.info("Saving final checkpoint...")
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
            'config_file': args.config if 'args' in locals() else 'unknown',
            'config': config.to_dict() if config else None,
            'error': str(e),
            'stack_trace': traceback.format_exc()
        }
        
        # Always use print for error logging to ensure output
        print("\nERROR: Training failed", file=sys.stderr)
        print(f"Error context: {error_context}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        
        # Try to use logger if available
        if logger and hasattr(logger, 'error'):
            try:
                logger.error(
                    "Training failed",
                    exc_info=True,
                    extra=error_context
                )
            except:
                pass  # Ignore logger errors in error handling
                
        sys.exit(1)

if __name__ == "__main__":
    main()

