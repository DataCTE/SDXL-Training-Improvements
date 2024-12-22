import logging
import traceback
import time
import psutil
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
from ..utils.paths import convert_windows_path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models import StableDiffusionXLModel, ModelType
from src.data.config import Config
from src.data.preprocessing.cache_manager import CacheManager
from src.core.memory.tensor import (
    tensors_to_device_,
    create_stream_context, 
    tensors_record_stream,
    torch_gc,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_
)

logger = logging.getLogger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer, 
        text_encoder_one: CLIPTextModel,
        text_encoder_two: CLIPTextModelWithProjection,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        use_cache: bool = True,
        max_retries: int = 3,
        chunk_size: int = 1000,
        max_memory_usage: float = 0.8
    ):
        super().__init__()
        self.config = config
        self.model = StableDiffusionXLModel(model_type=ModelType.BASE)
        self.model.tokenizer_1 = tokenizer_one
        self.model.tokenizer_2 = tokenizer_two
        self.model.text_encoder_1 = text_encoder_one 
        self.model.text_encoder_2 = text_encoder_two
        self.model.vae = vae
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
        proportion_empty_prompts: float = 0,
        batch_size: int = 64
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts with optimized memory handling."""
        try:
            # Tokenize prompts
            tokens_1 = []
            tokens_2 = []
            
            for i in range(0, len(prompt_batch), batch_size):
                batch = prompt_batch[i:i + batch_size]
                
                # Use high-level SDXL model interface
                with torch.no_grad():
                    text_encoder_output, pooled_output = self.model.encode_text(
                        train_device=self.device,
                        batch_size=len(batch),
                        text=batch,
                        text_encoder_1_layer_skip=0,
                        text_encoder_2_layer_skip=0,
                        text_encoder_1_output=None,
                        text_encoder_2_output=None
                    )

            return {
                "prompt_embeds": text_encoder_output,
                "pooled_prompt_embeds": pooled_output
            }

        except Exception as e:
            logger.error(f"Failed to encode prompts: {str(e)}")
            raise
            
        finally:
            torch_gc()

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Encode images with optimized memory handling."""
        try:
            if not isinstance(pixel_values, torch.Tensor):
                pixel_values = torch.stack(list(pixel_values))
                
            # Pin memory for faster host-device transfer    
            pin_tensor_(pixel_values)
            
            latents = []
            for idx in range(0, len(pixel_values), batch_size):
                batch = pixel_values[idx:idx + batch_size]
                
                # Create stream for pipelined processing
                with create_stream_context() as (compute_stream,):
                    with compute_stream:
                        # Move to device if needed
                        if not device_equals(batch.device, self.device):
                            tensors_to_device_([batch], self.device)
                            
                        # Use optimized VAE encoder
                        batch_latents = self.vae_encoder.encode(batch, return_dict=False)
                        
                        # Pin and record stream
                        pin_tensor_(batch_latents)
                        tensors_record_stream(compute_stream, batch_latents)
                        
                        # Store result
                        latents.append(batch_latents.cpu())
                        
                        # Clean up
                        unpin_tensor_(batch_latents)
                        
                        # Log memory stats
                        if self.enable_memory_tracking:
                            stats = self.vae_encoder.get_memory_stats()
                            logger.debug(f"VAE encoder memory stats: {stats}")
                        
                        # Replace tensors to reuse memory
                        if latents:
                            replace_tensors_(latents[-1], batch_latents)
                            
            # Combine results            
            latents = torch.cat(latents)
            
            return {"model_input": latents}
            
        except Exception as e:
            logger.error(f"Failed to encode images: {str(e)}")
            raise
            
        finally:
            # Clean up
            unpin_tensor_(pixel_values)
            torch_gc()
