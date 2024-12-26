"""High-level latent preprocessing with optimized orchestration."""
from typing import Dict, List, Optional, Union
import sys
import time
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
        # Add embedding processor
        self.embedding_processor = sdxl_model.embedding_processor
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
            
            # Use CLIP encoder's embedding processor
            self.embedding_processor = sdxl_model.clip_encoder_1
            
            # Initialize VAE encoder
            self.vae_encoder = VAEEncoder(
                vae=sdxl_model.vae,
                device=self.device,
                dtype=next(sdxl_model.vae.parameters()).dtype
            )
            
            # Use model's CLIP encoders
            self.clip_encoder_1 = sdxl_model.clip_encoder_1
            self.clip_encoder_2 = sdxl_model.clip_encoder_2
            
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
        """Enhanced prompt encoding with embedding processing.
        
        Args:
            prompt_batch: List of text prompts to encode
            
        Returns:
            Dictionary containing text embeddings and metadata
        """
        try:
            # Process prompts with embedding processor
            processed_prompts = []
            for prompt in prompt_batch:
                processed = self.embedding_processor.process_embeddings(prompt)
                processed_prompts.append(processed["processed_text"])

            # Get embeddings from both encoders
            encoder_1_output = self.clip_encoder_1.encode_prompt(processed_prompts)
            encoder_2_output = self.clip_encoder_2.encode_prompt(processed_prompts)

            # Combine into consolidated format
            text_latent = {
                "embeddings": {
                    "prompt_embeds": encoder_1_output["text_embeds"],
                    "pooled_prompt_embeds": encoder_1_output["pooled_embeds"],
                    "prompt_embeds_2": encoder_2_output["text_embeds"],
                    "pooled_prompt_embeds_2": encoder_2_output["pooled_embeds"]
                },
                "caption": prompt_batch[0] if prompt_batch else "",
                "processed_text": processed_prompts[0] if processed_prompts else "",
                "metadata": {
                    "num_prompts": len(prompt_batch),
                    "device": str(self.device),
                    "dtype": str(next(iter(encoder_1_output["text_embeds"])).dtype),
                    "timestamp": time.time(),
                    "embedding_shapes": {
                        k: tuple(v.shape) 
                        for k, v in encoder_1_output["text_embeds"].items()
                    }
                }
            }

            return text_latent
            
        except Exception as e:
            logger.error("Failed to encode prompts", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'batch_size': len(prompt_batch),
                'stack_trace': True
            })
            raise

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images using VAE with robust error handling.
        
        Args:
            pixel_values: Image tensor in [0, 1] range, shape (B, C, H, W)
            
        Returns:
            Dictionary containing encoded latents
        """
        try:
            # Use the VAE encoder's encode method directly
            encoded_output = self.vae_encoder.encode(
                pixel_values=pixel_values,
                num_images_per_prompt=1,
                output_hidden_states=False
            )
            
            # Extract latents from the output
            latents = encoded_output.get("latent_dist")
            if latents is None:
                raise ValueError("VAE encoder did not return latent distribution")
            
            # Check for NaN/Inf values with detailed location tracking
            nan_mask = torch.isnan(latents)
            inf_mask = torch.isinf(latents)
        
            if nan_mask.any() or inf_mask.any():
                nan_count = nan_mask.sum().item()
                inf_count = inf_mask.sum().item()
            
                # Get indices of first few NaN/Inf values for debugging
                nan_indices = torch.where(nan_mask)
                inf_indices = torch.where(inf_mask)
            
                error_context = {
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'total_elements': latents.numel(),
                    'nan_percentage': (nan_count / latents.numel()) * 100,
                    'first_nan_indices': [
                        tuple(idx[i].item() for idx in nan_indices)
                        for i in range(min(5, len(nan_indices[0])))
                    ] if nan_count > 0 else [],
                    'first_inf_indices': [
                        tuple(idx[i].item() for idx in inf_indices)
                        for i in range(min(5, len(inf_indices[0])))
                    ] if inf_count > 0 else [],
                    'input_shape': tuple(pixel_values.shape),
                    'latent_shape': tuple(latents.shape)
                }
            
                error_msg = (
                    f"VAE produced invalid values: {nan_count} NaN, {inf_count} Inf. "
                    f"First NaN at: {error_context['first_nan_indices']}, "
                    f"First Inf at: {error_context['first_inf_indices']}"
                )
                logger.error(error_msg, extra=error_context)
                # Force script termination
                sys.exit(1)

            # Return in expected format for cache manager
            return {
                "image_latent": latents,
                "metadata": {
                    "shape": tuple(latents.shape),
                    "dtype": str(latents.dtype),
                    "device": str(latents.device),
                    "stats": {
                        "min": latents.min().item(),
                        "max": latents.max().item(),
                        "mean": latents.mean().item(),
                        "std": latents.std().item()
                    }
                }
            }

        except Exception as e:
            logger.error("VAE encoding failed", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
                'stack_trace': True
            })
            raise
