import logging
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
from src.data.utils.paths import convert_windows_path
import torch
from src.models import StableDiffusionXLModel
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager


logger = logging.getLogger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        sdxl_model: StableDiffusionXLModel,
        device: Union[str, torch.device] = "cuda",  # This is correct
        max_retries: int = 3,
        chunk_size: int = 1000,
        max_memory_usage: float = 0.8
    ):
        """Initialize the LatentPreprocessor.

        Args:
            config (Config): Configuration object containing cache settings
            sdxl_model (StableDiffusionXLModel): SDXL model instance
            device (Union[str, torch.device], optional): Device to run operations on. Defaults to "cuda".
            max_retries (int, optional): Maximum number of retries for operations. Defaults to 3.
            chunk_size (int, optional): Size of processing chunks. Defaults to 1000.
            max_memory_usage (float, optional): Maximum memory usage fraction. Defaults to 0.8.
        """
        self.config = config
        self.model = sdxl_model
        # Convert string to device object if needed
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        
        # Setup cache after initializing attributes
        self._setup_cache(config)

    def _setup_cache(self, config: Config) -> None:
        """Setup caching configuration."""
        self.use_cache = config.global_config.cache.use_cache
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

    def encode_prompt(self, prompt_batch: List[str]) -> Dict[str, torch.Tensor]:
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

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images using SDXL VAE."""
        try:
            with torch.no_grad():
                return {"model_input": self.model.vae.encode(pixel_values.to(self.device)).latent_dist.sample()}
        except Exception as e:
            logger.error(f"Failed to encode images: {str(e)}")
            raise
