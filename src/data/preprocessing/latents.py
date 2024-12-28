from typing import Dict, List, Union
import sys
import time
import torch
from pathlib import Path

from src.core.logging import get_logger, LogConfig
from src.models.encoders.vae import VAEEncoder
from src.models.encoders.vae import VAEEncoder 
from src.models.encoders.clip import CLIPEncoder
from src.models import StableDiffusionXLModel
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager
from src.core.types import DataType, ModelWeightDtypes
from src.data.utils.paths import convert_windows_path

logger = get_logger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        sdxl_model: StableDiffusionXLModel,
        device: Union[str, torch.device] = "cuda",
        max_retries: int = 3,
        chunk_size: int = 1000,
        max_memory_usage: float = 0.8
    ):
        try:
            self.config = config
            self.model = sdxl_model
            self.device = torch.device(device) if isinstance(device, str) else device
            
            # Validate model components are initialized
            if not sdxl_model.vae:
                raise ValueError("VAE is not initialized in SDXL model")
                
            # Initialize VAE encoder in float32
            self.vae_encoder = VAEEncoder(
                vae=sdxl_model.vae,
                device=self.device,
                dtype=torch.float32
            )
            
            # Validate CLIP encoders
            if not sdxl_model.clip_encoder_1:
                raise ValueError("CLIP encoder 1 is not initialized in SDXL model")
            if not sdxl_model.clip_encoder_2:
                raise ValueError("CLIP encoder 2 is not initialized in SDXL model")
                
            # CLIP encoders
            self.clip_encoder_1 = sdxl_model.clip_encoder_1
            self.clip_encoder_2 = sdxl_model.clip_encoder_2
            
            # Set embedding processor and validate
            self.embedding_processor = sdxl_model.clip_encoder_1
            if not self.embedding_processor:
                raise ValueError("Embedding processor could not be initialized")
                
            # Validate embedding processor has required method
            if not hasattr(self.embedding_processor, 'process_embeddings'):
                raise ValueError("Embedding processor missing process_embeddings method")
                
            self.max_retries = max_retries
            self.chunk_size = chunk_size
            self.max_memory_usage = max_memory_usage
            
            self._setup_cache(config)
            
            logger.info("Initialized latent preprocessor", extra={
                'device': str(self.device),
                'vae_initialized': self.vae_encoder is not None,
                'clip1_initialized': self.clip_encoder_1 is not None,
                'clip2_initialized': self.clip_encoder_2 is not None,
                'embedding_processor_initialized': self.embedding_processor is not None
            })
            
        except Exception as e:
            logger.error("Failed to initialize latent preprocessor", 
                        extra={
                            'error': str(e),
                            'error_type': type(e).__name__,
                            'device': str(device),
                            'model_type': type(sdxl_model).__name__,
                            'vae_present': hasattr(sdxl_model, 'vae') and sdxl_model.vae is not None,
                            'clip1_present': hasattr(sdxl_model, 'clip_encoder_1') and sdxl_model.clip_encoder_1 is not None,
                            'clip2_present': hasattr(sdxl_model, 'clip_encoder_2') and sdxl_model.clip_encoder_2 is not None
                        })
            raise
            
    def _validate_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """Validate and clean tensor values."""
        if tensor is None:
            raise ValueError(f"Tensor {name} is None")
            
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0)
            logger.warning(f"Fixed NaN values in {name}")
            
        if torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
            logger.warning(f"Fixed Inf values in {name}")
            
        return tensor

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images with improved validation."""
        try:
            # Validate input
            pixel_values = self._validate_tensor(pixel_values, "input")
            pixel_values = pixel_values.to(dtype=torch.float32)
            
            # Get VAE encoding with retries
            for attempt in range(self.max_retries):
                try:
                    encoded_output = self.vae_encoder.encode(
                        pixel_values=pixel_values,
                        num_images_per_prompt=1,
                        output_hidden_states=False
                    )
                    
                    latents = encoded_output["latent_dist"]
                    uncond_latents = encoded_output["uncond_latents"]
                    
                    # Validate outputs
                    latents = self._validate_tensor(latents, "latents")
                    uncond_latents = self._validate_tensor(uncond_latents, "uncond_latents")
                    
                    return {
                        "image_latent": latents,
                        "uncond_latents": uncond_latents,
                        "metadata": {
                            "input_shape": tuple(pixel_values.shape),
                            "latent_shape": tuple(latents.shape),
                            "scaling_factor": 0.18215,
                            "stats": {
                                "min": latents.min().item(),
                                "max": latents.max().item(),
                                "mean": latents.mean().item(),
                                "std": latents.std().item()
                            }
                        }
                    }
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error("VAE encoding failed", extra={
                'error': str(e),
                'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
            })
            raise

    def encode_prompt(self, prompt_batch: List[str]) -> Dict[str, torch.Tensor]:
        try:
            # Process prompts
            processed_prompts = [
                self.embedding_processor.process_embeddings(prompt)["processed_text"]
                for prompt in prompt_batch
            ]

            # Get embeddings with validation
            encoder_1_output = self.clip_encoder_1.encode_prompt(processed_prompts)
            encoder_2_output = self.clip_encoder_2.encode_prompt(processed_prompts)
            
            # Validate embeddings
            embeds_1 = self._validate_tensor(encoder_1_output["text_embeds"], "text_embeds_1")
            pooled_1 = self._validate_tensor(encoder_1_output["pooled_embeds"], "pooled_embeds_1")
            embeds_2 = self._validate_tensor(encoder_2_output["text_embeds"], "text_embeds_2")
            pooled_2 = self._validate_tensor(encoder_2_output["pooled_embeds"], "pooled_embeds_2")

            return {
                "embeddings": {
                    "prompt_embeds": embeds_1,
                    "pooled_prompt_embeds": pooled_1,
                    "prompt_embeds_2": embeds_2,
                    "pooled_prompt_embeds_2": pooled_2
                },
                "caption": prompt_batch[0] if prompt_batch else "",
                "processed_text": processed_prompts[0] if processed_prompts else "",
                "metadata": {
                    "num_prompts": len(prompt_batch),
                    "device": str(self.device),
                    "dtype": str(embeds_1.dtype),
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
           
            logger.error("Failed to encode prompts", 
                        extra={'error': str(e), 'batch_size': len(prompt_batch)})
            raise
            
    def _setup_cache(self, config: Config) -> None:
        """Setup caching system."""
        self.use_cache = config.global_config.cache.use_cache
        if not self.use_cache:
            return
            
        try:
            cache_dir = Path(convert_windows_path(
                config.global_config.cache.cache_dir,
            ))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            model_dtypes = ModelWeightDtypes(
                train_dtype=DataType.from_str(config.model.dtype),
                fallback_train_dtype=DataType.from_str(config.model.fallback_dtype),
                unet=DataType.from_str(config.model.unet_dtype or config.model.dtype),
                prior=DataType.from_str(config.model.prior_dtype or config.model.dtype),
                text_encoder=DataType.from_str(config.model.text_encoder_dtype or config.model.dtype),
                text_encoder_2=DataType.from_str(config.model.text_encoder_2_dtype or config.model.dtype),
                vae=DataType.from_str("float32"),
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
