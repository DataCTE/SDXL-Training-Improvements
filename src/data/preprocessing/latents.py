import logging
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
from src.data.utils.paths import convert_windows_path
import torch
from src.models import StableDiffusionXLModel
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager
from src.core.types import DataType, ModelWeightDtypes


logger = logging.getLogger(__name__)


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
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        self.config = config
        self.model = sdxl_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        self._setup_cache(config)

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
        try:
            with torch.no_grad():
                txt_out, pooled_out = self.model.encode_text(
                    train_device=self.device,
                    batch_size=len(prompt_batch),
                    text=prompt_batch,
                    text_encoder_1_layer_skip=0,
                    text_encoder_2_layer_skip=0
                )
            return {"prompt_embeds": txt_out, "pooled_prompt_embeds": pooled_out}
        except Exception as e:
            logger.error(f"Failed to encode prompts: {str(e)}")
            raise

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            with torch.no_grad():
                # Ensure pixel_values has the same dtype as the VAE
                vae_dtype = next(self.model.vae.parameters()).dtype
                pixel_values = pixel_values.to(dtype=vae_dtype)
                latents = self.model.vae.encode(pixel_values).latents
                latents = latents * self.model.vae.config.scaling_factor
            return {"latent": latents}
        except Exception as e:
            logger.error(f"Failed to encode images: {str(e)}")
            raise
