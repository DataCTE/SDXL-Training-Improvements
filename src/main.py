"""Main orchestration script for SDXL fine-tuning with 100x speedups."""
import argparse
import os
import time

# Disable TorchDynamo in DataLoader workers
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import multiprocessing as mp
import torch.cuda
import logging
import sys
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from torch.distributed import init_process_group
from src.training.optimizers import AdamWScheduleFreeKahan
from src.training.optimizers import AdamWBF16
from src.training.optimizers import SOAP
from src.core.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.core.logging import setup_logging
from src.core.logging.wandb import WandbLogger
from src.core.types import DataType, ModelWeightDtypes
from src.data.preprocessing import CacheManager, PreprocessingPipeline, create_tag_weighter_with_index
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

def setup_device_and_logging(config: Config) -> torch.device:
    # First, clear all existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    
    # Create console handler for root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set up file logging
    output_dir = Path(config.global_config.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Explicitly set DDPM trainer logger level
    ddpm_logger = logging.getLogger("src.training.methods.ddpm_trainer")
    ddpm_logger.setLevel(logging.DEBUG)
    
    # Add verification logs
    root_logger.debug("Root logger initialized at DEBUG level")
    ddpm_logger.debug("DDPM trainer logger initialized at DEBUG level")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    if is_main_process():
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(device.index)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.1f} GB")
    return device

def setup_model(config: Config, device: torch.device) -> Optional[StableDiffusionXLModel]:
    """Initialize SDXL model components."""
    logger.info("Loading models...")
    model = None  # Initialize model variable outside try block
    try:
        # Create model instance
        model = StableDiffusionXLModel(
            ModelType.BASE,
            enable_memory_efficient_attention=config.model.enable_memory_efficient_attention,
            enable_vae_slicing=config.model.enable_vae_slicing,
            enable_model_cpu_offload=config.model.enable_model_cpu_offload
        )

        # Load pretrained components
        logger.info(f"Loading pretrained model from {config.model.pretrained_model_name}")
        model.from_pretrained(
            config.model.pretrained_model_name,
            dtype=config.model.dtype,
            use_safetensors=True
        )

        # Move model to device after loading
        logger.info(f"Moving model to {device}")
        model.to(device)
        
        # Verify model components
        if not all([
            model.vae,
            model.text_encoder_1,
            model.text_encoder_2,
            model.tokenizer_1,
            model.tokenizer_2,
            model.unet,
            model.noise_scheduler,
            model.clip_encoder_1,
            model.clip_encoder_2,
            model.vae_encoder
        ]):
            missing = []
            if not model.vae: missing.append("VAE")
            if not model.text_encoder_1: missing.append("Text Encoder 1") 
            if not model.text_encoder_2: missing.append("Text Encoder 2")
            if not model.tokenizer_1: missing.append("Tokenizer 1")
            if not model.tokenizer_2: missing.append("Tokenizer 2")
            if not model.unet: missing.append("UNet")
            if not model.noise_scheduler: missing.append("Noise Scheduler")
            if not model.clip_encoder_1: missing.append("CLIP Encoder 1")
            if not model.clip_encoder_2: missing.append("CLIP Encoder 2")
            if not model.vae_encoder: missing.append("VAE Encoder")
            
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
            'model_state': 'partially_loaded' if model else 'not_created'
        }
        logger.error("Failed to initialize model", extra=error_context)
        raise RuntimeError(f"Failed to load models: {str(e)}") from e


def load_training_data(config: Config) -> tuple[List[str], List[str]]:
    image_paths, captions = [], []
    train_dirs = (config.data.train_data_dir if isinstance(config.data.train_data_dir, list) 
                  else [config.data.train_data_dir])
    train_dirs = convert_path_list(train_dirs)
    for data_dir in train_dirs:
        dir_path = Path(data_dir)
        logger.info(f"Processing dataset directory: {dir_path}")
        if not dir_path.exists():
            logger.warning(f"Training directory not found: {data_dir}")
            continue
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        dir_images = []
        for ext in image_extensions:
            dir_images.extend(list(dir_path.glob(ext)))
        if not dir_images:
            logger.warning(f"No images found in {data_dir} (supported: {', '.join(image_extensions)})")
            continue
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

def setup_training(
    config: Config,
    model: StableDiffusionXLModel,
    device: torch.device,
    image_paths: List[str],
    captions: List[str]
) -> tuple[torch.utils.data.DataLoader, torch.optim.Optimizer, Optional[WandbLogger]]:
    try:
        if torch.cuda.is_available():
            logger.info(f"Initial CUDA memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
        logger.info("Initializing latent preprocessor...")
        try:
            latent_preprocessor = LatentPreprocessor(config=config, sdxl_model=model, device=device)
            logger.info("Latent preprocessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize latent preprocessor: {str(e)}", exc_info=True)
            raise
        
        logger.info("Setting up cache manager...")
        try:
            model_dtypes = ModelWeightDtypes(
                train_dtype=DataType.from_str(config.model.dtype),
                fallback_train_dtype=DataType.from_str(config.model.fallback_dtype),
                unet=DataType.from_str(config.model.unet_dtype or config.model.dtype),
                prior=DataType.from_str(config.model.prior_dtype or config.model.dtype),
                text_encoder=DataType.from_str(config.model.text_encoder_dtype or config.model.dtype),
                text_encoder_2=DataType.from_str(config.model.text_encoder_2_dtype or config.model.dtype),
                vae=DataType.from_str(config.model.vae_dtype or config.model.dtype),
                effnet_encoder=DataType.from_str(config.model.effnet_dtype or config.model.dtype),
                decoder=DataType.from_str(config.model.decoder_dtype or config.model.dtype),
                decoder_text_encoder=DataType.from_str(config.model.decoder_text_encoder_dtype or config.model.dtype),
                decoder_vqgan=DataType.from_str(config.model.decoder_vqgan_dtype or config.model.dtype),
                lora=DataType.from_str(config.model.lora_dtype or config.model.dtype),
                embedding=DataType.from_str(config.model.embedding_dtype or config.model.dtype)
            )
            cache_manager = CacheManager(
                model_dtypes=model_dtypes,
                cache_dir=Path(convert_windows_path(config.global_config.cache.cache_dir)),
                num_proc=config.global_config.cache.num_proc,
                chunk_size=config.global_config.cache.chunk_size,
                compression=getattr(config.global_config.cache, 'compression', 'zstd'),
                verify_hashes=config.global_config.cache.verify_hashes,
                max_memory_usage=0.8,
                enable_memory_tracking=True
            )
            cache_manager.validate_cache_index()
            logger.info("Cache manager setup complete")
        except Exception as e:
            logger.error(f"Failed to setup cache manager: {str(e)}", exc_info=True)
            raise
        
        logger.info("Creating preprocessing pipeline...")
        try:
            preprocessing_pipeline = PreprocessingPipeline(
                config=config,
                latent_preprocessor=latent_preprocessor,
                cache_manager=cache_manager,
                is_train=True,
                num_gpu_workers=config.preprocessing.num_gpu_workers,
                num_cpu_workers=config.preprocessing.num_cpu_workers,
                num_io_workers=config.preprocessing.num_io_workers,
                prefetch_factor=config.preprocessing.prefetch_factor,
                use_pinned_memory=False  # Disable pinned memory to prevent CUDA tensor pinning issues
            )
        except Exception as e:
            logger.error(f"Failed to create preprocessing pipeline: {str(e)}", exc_info=True)
            raise
        
        # Create tag weighter with proper configuration
        if config.tag_weighting.enable_tag_weighting:
            logger.info("Initializing tag weighter...")
            tag_weighter = create_tag_weighter_with_index(
                config=config,
                image_captions={path: caption for path, caption in zip(image_paths, captions)},
                index_output_path=Path(config.global_config.output_dir) / "tag_weights.json",
                cache_path=Path(config.global_config.cache.cache_dir) / "tag_weights_cache.json"
            )
        else:
            tag_weighter = None

        logger.info("Creating dataset...")
        try:
            train_dataset = create_dataset(
                config=config,
                image_paths=image_paths,
                captions=captions,
                preprocessing_pipeline=preprocessing_pipeline,
                tag_weighter=tag_weighter,
                enable_memory_tracking=True,
                max_memory_usage=0.8
            )
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}", exc_info=True)
            raise
        
        logger.info("Creating dataloader...")
        try:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.training.batch_size // config.training.gradient_accumulation_steps,
                shuffle=True,
                num_workers=0,  # Use single worker to avoid pickling issues
                pin_memory=False,  # Disable pinned memory to prevent CUDA tensor pinning issues
                persistent_workers=False,  # Disable persistent workers
                collate_fn=train_dataset.collate_fn
            )
        except Exception as e:
            logger.error(f"Failed to create dataloader: {str(e)}", exc_info=True)
            raise
        
        logger.info("Setting up optimizer...")
        try:
            optimizer_kwargs = {
                "lr": config.training.learning_rate,
                "betas": config.training.optimizer_betas,
                "weight_decay": config.training.weight_decay,
                "eps": config.training.optimizer_eps
            }

            # Select optimizer based on config and hardware capabilities
            if config.model.enable_bf16_training and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                optimizer = AdamWBF16(model.unet.parameters(), **optimizer_kwargs)
                logger.info("Using BF16-optimized AdamW optimizer")
            elif config.training.method == "flow_matching":
                optimizer = SOAP(
                    model.unet.parameters(),
                    **optimizer_kwargs,
                    precondition_frequency=10,
                    max_precond_dim=10000
                )
                logger.info("Using SOAP optimizer for flow matching")
            elif config.training.zero_terminal_snr:
                optimizer = AdamWScheduleFreeKahan(
                    model.unet.parameters(),
                    **optimizer_kwargs,
                    warmup_steps=config.training.warmup_steps,
                    kahan_sum=True
                )
                logger.info("Using Schedule-free AdamW with Kahan summation")
            else:
                optimizer = torch.optim.AdamW(model.unet.parameters(), **optimizer_kwargs)
                logger.info("Using standard AdamW optimizer")
        except Exception as e:
            logger.error(f"Failed to setup optimizer: {str(e)}", exc_info=True)
            raise

        # Add timeout handling
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not torch.cuda.is_available() or torch.cuda.current_stream().query():
                break
            time.sleep(0.1)
            
        if time.time() - start_time >= timeout:
            raise TimeoutError("CUDA operations timed out")

        # Force CUDA synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info("CUDA synchronized successfully")

        try:
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                model.unet = torch.nn.parallel.DistributedDataParallel(
                    model.unet, device_ids=[device] if device.type == "cuda" else None
                )
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {str(e)}", exc_info=True)
            raise

        wandb_logger = None
        if config.training.use_wandb and is_main_process():
            try:
                wandb_logger = WandbLogger(
                    project="sdxl-training",
                    name=Path(config.global_config.output_dir).name,
                    config=config.__dict__,
                    dir=config.global_config.output_dir,
                    tags=["sdxl", "fine-tuning"]
                )
                # Explicitly log model parameters
                wandb_logger.log_model(model.unet)
            except Exception as e:
                logger.warning(f"Failed to initialize WandB logging: {str(e)}", exc_info=True)
                wandb_logger = None

        return train_dataloader, optimizer, wandb_logger

    except TimeoutError:
        logger.error("Setup timed out after 5 minutes", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in setup_training: {str(e)}", exc_info=True)
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

def main():
    print("Starting SDXL training script...")
    mp.set_start_method('spawn', force=True)
    
    # Check and configure system limits
    check_system_limits()
    
    device = None
    try:
        args = parse_args()
        config = Config.from_yaml(args.config)
        
        with setup_environment(args):
            device = setup_device_and_logging(config)
            model = setup_model(config, device)
            
            if model is None:
                raise RuntimeError(f"Failed to initialize model from {config.model.pretrained_model_name}")
            
            logger.info("Model initialized successfully")
            
            # Move model to device if needed
            if hasattr(model, 'state_dict'):
                try:
                    if not tensors_match_device(model.state_dict(), device):
                        logger.info(f"Moving model to device {device}")
                        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                        pre_transfer_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        
                        with create_stream_context(stream):
                            tensors_to_device_(model.state_dict(), device)
                            if stream:
                                stream.synchronize()
                            torch_sync()
                            
                        if torch.cuda.is_available():
                            post_transfer_memory = torch.cuda.memory_allocated()
                            memory_delta = post_transfer_memory - pre_transfer_memory
                            logger.debug(f"Memory delta: {memory_delta / 1024**2:.1f}MB")
                except Exception as e:
                    raise TrainingSetupError(
                        "Failed to move model to device",
                        {
                            "error": str(e),
                            "device": str(device),
                            "cuda_available": torch.cuda.is_available(),
                            "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        }
                    )

            # Setup training components
            setup_memory_optimizations(model.unet, config, device)
            if is_main_process():
                verify_memory_optimizations(model.unet, config, device, logger)
                
            logger.info("Loading training data...")
            image_paths, captions = load_training_data(config)
            
            logger.info("Setting up training...")
            train_dataloader, optimizer, wandb_logger = setup_training(
                config=config,
                model=model,  # Fixed: was using undefined 'models'
                device=device,
                image_paths=image_paths,
                captions=captions
            )
            
            trainer = create_trainer(
                config=config,
                model=model,  # Fixed: was using undefined 'models'
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
        error_context = {
            'error_type': type(e).__name__,
            'device': str(device) if device is not None else 'unknown',
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Add config-related info if available
        if 'config' in locals():
            error_context['model_name'] = config.model.pretrained_model_name
            
        # Add CUDA memory info if available
        if torch.cuda.is_available():
            error_context.update({
                'cuda_memory_allocated': torch.cuda.memory_allocated(),
                'cuda_memory_reserved': torch.cuda.memory_reserved(),
                'cuda_max_memory': torch.cuda.max_memory_allocated()
            })
            
        logger.error(f"Training failed: {str(e)}", extra=error_context)
        
        if isinstance(e, TrainingSetupError) and e.context:
            logger.error(f"Error context: {e.context}")
            
        sys.exit(1)

if __name__ == "__main__":
    main()

