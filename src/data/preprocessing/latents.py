"""High-level latent preprocessing with optimized orchestration."""
from typing import Dict, List, Optional, Union
import torch
from pathlib import Path

from src.core.logging.logging import setup_logging
from src.models.encoders.vae import VAEEncoder
from src.models.encoders.clip import CLIPEncoder
from src.models import StableDiffusionXLModel
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager
from src.core.types import DataType, ModelWeightDtypes
from src.data.utils.paths import convert_windows_path

# Initialize logger with core logging system
logger = setup_logging(__name__)


class LatentPreprocessor:
    """High-level orchestrator for latent preprocessing using optimized encoders."""
    
    def __init__(
        self,
        config: Config,
        sdxl_model: StableDiffusionXLModel,
        device: Union[str, torch.device] = "cuda",
        max_retries: int = 3,
        chunk_size: int = 1000,
        max_memory_usage: float = 0.8
    ):
        """Initialize preprocessor with model and encoders.
        
        Args:
            config: Configuration object
            sdxl_model: SDXL model instance
            device: Target device
            max_retries: Maximum retry attempts
            chunk_size: Processing chunk size
            max_memory_usage: Maximum memory usage fraction
        """
        try:
            self.config = config
            self.model = sdxl_model
            self.device = torch.device(device) if isinstance(device, str) else device
            
            # Initialize VAE encoder
            self.vae_encoder = VAEEncoder(
                vae=sdxl_model.vae,
                device=self.device,
                dtype=next(sdxl_model.vae.parameters()).dtype
            )
            
            # Initialize CLIP encoders
            self.clip_encoder_1 = CLIPEncoder(
                text_encoder=sdxl_model.text_encoder_1,
                tokenizer=sdxl_model.tokenizer_1,
                device=self.device,
                dtype=next(sdxl_model.text_encoder_1.parameters()).dtype
            )
            
            self.clip_encoder_2 = CLIPEncoder(
                text_encoder=sdxl_model.text_encoder_2,
                tokenizer=sdxl_model.tokenizer_2,
                device=self.device,
                dtype=next(sdxl_model.text_encoder_2.parameters()).dtype
            )
            
            self.max_retries = max_retries
            self.chunk_size = chunk_size
            self.max_memory_usage = max_memory_usage
            
            # Setup cache if enabled
            self._setup_cache(config)
            
            logger.info("Latent preprocessor initialized", extra={
                'device': str(self.device),
                'model_type': type(self.model).__name__,
                'config': {
                    'dtype': config.model.dtype,
                    'cache_enabled': config.global_config.cache.use_cache,
                    'max_retries': max_retries,
                    'chunk_size': chunk_size
                }
            })
            
        except Exception as e:
            logger.error("Failed to initialize latent preprocessor", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'device': str(device),
                'stack_trace': True
            })
            raise

    def _setup_cache(self, config: Config) -> None:
        self.use_cache = config.global_config.cache.use_cache
        if not self.use_cache:
            return
        try:
            cache_dir = Path(convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True))
            cache_dir.mkdir(parents=True, exist_ok=True)
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
            
            self.cache_manager = CacheManager(
                model_dtypes=model_dtypes,
                cache_dir=cache_dir,
                num_proc=config.global_config.cache.num_proc,
                chunk_size=config.global_config.cache.chunk_size,
                compression=getattr(config.global_config.cache, 'compression', 'zstd'),
                verify_hashes=config.global_config.cache.verify_hashes,
                max_memory_usage=self.max_memory_usage
            )
        except Exception as e:
            logger.error(f"Failed to setup cache: {str(e)}")
            self.use_cache = False

    def encode_prompt(self, prompt_batch: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts using both CLIP encoders."""
        try:
            # Get embeddings from both encoders
            encoder_1_output = self.clip_encoder_1.encode_prompt(prompt_batch)
            encoder_2_output = self.clip_encoder_2.encode_prompt(prompt_batch)
            
            # Combine results
            return {
                "prompt_embeds": encoder_1_output["text_embeds"],
                "pooled_prompt_embeds": encoder_1_output["pooled_embeds"],
                "prompt_embeds_2": encoder_2_output["text_embeds"],
                "pooled_prompt_embeds_2": encoder_2_output["pooled_embeds"]
            }
            
        except Exception as e:
            logger.error("Failed to encode prompts", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'batch_size': len(prompt_batch),
                'stack_trace': True
            })
            raise

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images using VAE encoder.
        
        Args:
            pixel_values: Image tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing encoded latents
        """
        try:
            logger.debug("Starting image encoding", extra={
                'input_shape': tuple(pixel_values.shape),
                'device': str(self.device)
            })
            
            # Use VAE encoder implementation
            result = self.vae_encoder.encode(pixel_values)
            
            logger.debug("Image encoding complete", extra={
                'output_shape': tuple(result["latent_dist"].shape)
            })
            
            return {"image_latent": result["latent_dist"]}
            
        except Exception as e:
            logger.error("Failed to encode images", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
                'stack_trace': True
            })
            raise
