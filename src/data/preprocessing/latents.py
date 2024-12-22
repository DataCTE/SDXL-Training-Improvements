import logging
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
from ..utils.paths import convert_windows_path
import torch
from src.models import StableDiffusionXLModel
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager
from src.core.memory.tensor import (
    device_equals
)

logger = logging.getLogger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        sdxl_model: StableDiffusionXLModel,
        device: Union[str, torch.device] = "cuda",
        use_cache: bool = True,
        max_retries: int = 3,
        chunk_size: int = 1000,
        max_memory_usage: float = 0.8
    ):
        super().__init__()
        self.config = config
        self.model = sdxl_model
        self.model.to(device)
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        self.device = device
        
        # Setup cache after initializing attributes
        self._setup_cache(config, use_cache)
        
    def _setup_cache(self, config: Config, use_cache: bool = True) -> None:
        """Setup caching configuration."""
        self.use_cache = use_cache
        if not self.use_cache:
            return
            
        try:
            cache_dir = Path(convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.cache_manager = CacheManager(
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

    def encode_prompt(
        self,
        prompt_batch: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts using SDXL model interface."""
        try:
            with torch.no_grad():
                text_encoder_output, pooled_output = self.model.encode_text(
                    train_device=self.device,
                    batch_size=len(prompt_batch),
                    text=prompt_batch,
                    text_encoder_1_layer_skip=0,
                    text_encoder_2_layer_skip=0
                )

            return {
                "prompt_embeds": text_encoder_output,
                "pooled_prompt_embeds": pooled_output
            }

        except Exception as e:
            logger.error(f"Failed to encode prompts: {str(e)}")
            raise

    def encode_images(
        self,
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode images using SDXL VAE."""
        try:
            # Validate input
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError(f"Expected tensor input, got {type(pixel_values)}")
                
            if pixel_values.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got {pixel_values.dim()}D")
                
            # Move to correct device if needed
            if not device_equals(pixel_values.device, self.device):
                pixel_values = pixel_values.to(self.device)
                
            # Encode with VAE
            with torch.no_grad():
                latents = self.model.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.model.vae.config.scaling_factor

            return {"model_input": latents}

        except Exception as e:
            logger.error(f"Failed to encode images: {str(e)}")
            raise
