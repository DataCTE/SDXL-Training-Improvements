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
        """Enhanced prompt encoding with embedding processing."""
        try:
            # Process prompts with embedding processor
            processed_prompts = []
            for prompt in prompt_batch:
                processed = self.embedding_processor.process_embeddings(prompt)
                processed_prompts.append(processed["processed_text"])

            # Get embeddings from both encoders
            encoder_1_output = self.clip_encoder_1.encode_prompt(processed_prompts)
            encoder_2_output = self.clip_encoder_2.encode_prompt(processed_prompts)

            # Combine results
            embeddings = {
                "prompt_embeds": encoder_1_output["text_embeds"],
                "pooled_prompt_embeds": encoder_1_output["pooled_embeds"],
                "prompt_embeds_2": encoder_2_output["text_embeds"],
                "pooled_prompt_embeds_2": encoder_2_output["pooled_embeds"]
            }

            # Validate embeddings
            if not self.embedding_processor.validate_embeddings(embeddings):
                raise ValueError("Invalid embeddings detected")

            return embeddings
            
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
            # 1. Input validation and normalization
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Input must be a tensor")
                
            # Ensure input is in [0, 1] range
            if pixel_values.min() < -0.1 or pixel_values.max() > 1.1:
                logger.warning(
                    f"Input tensor out of range: min={pixel_values.min().item():.3f}, "
                    f"max={pixel_values.max().item():.3f}"
                )
                pixel_values = torch.clamp(pixel_values, 0, 1)

            # Convert to [-1, 1] range
            pixel_values = 2 * pixel_values - 1
                
            # 2. Proper dtype handling
            dtype = next(self.vae_encoder.vae.parameters()).dtype
            pixel_values = pixel_values.to(device=self.device, dtype=dtype)
                
            # 3. Process with error handling
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                # Get latent distribution
                vae_output = self.vae_encoder.vae.encode(pixel_values)
                    
                if hasattr(vae_output, 'latent_dist'):
                    # Sample from distribution with scaling
                    latents = vae_output.latent_dist.sample() * 0.18215
                else:
                    latents = vae_output.sample() * 0.18215
                        
                # 4. Validate outputs
                if torch.isnan(latents).any():
                    # Try to recover from NaNs
                    logger.warning("NaN values detected in latents, attempting recovery...")
                    latents = torch.nan_to_num(latents, nan=0.0)
                    latents = torch.clamp(latents, -1e6, 1e6)
                    
                    # Check if recovery worked
                    if torch.isnan(latents).any():
                        nan_count = torch.isnan(latents).sum().item()
                        nan_indices = torch.where(torch.isnan(latents))
                        raise ValueError(
                            f"VAE produced {nan_count} NaN values in latents. "
                            f"First NaN at index: {[idx[0].item() for idx in nan_indices]}"
                        )
                        
                if torch.isinf(latents).any():
                    latents = torch.clamp(latents, -1e6, 1e6)

                # Log successful encoding
                logger.debug("VAE encoding complete", extra={
                    'output_shape': tuple(latents.shape),
                    'output_dtype': str(latents.dtype),
                    'output_stats': {
                        'min': latents.min().item(),
                        'max': latents.max().item(),
                        'mean': latents.mean().item(),
                        'std': latents.std().item()
                    }
                })
                    
                return {"image_latent": latents}

        except Exception as e:
            logger.error("VAE encoding failed", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
                'stack_trace': True
            })
            raise
