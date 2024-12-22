import logging
import traceback
import time
import psutil
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from src.models.encoders.vae import VAEEncoder
from src.models.encoders.clip import encode_clip
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
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one 
        self.text_encoder_two = text_encoder_two
        # Initialize optimized VAE encoder
        self.vae_encoder = VAEEncoder(
            vae=vae,
            device=device,
            enable_memory_efficient_attention=True,
            enable_vae_slicing=True,
            enable_vae_tiling=False,
            vae_tile_size=512,
            enable_gradient_checkpointing=True
        )
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
            cache_dir = Path(config.global_config.cache.cache_dir)
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
                
                # Process tokenization in parallel streams
                with create_stream_context() as (stream_1, stream_2):
                    # Tokenize first encoder
                    with stream_1:
                        batch_tokens_1 = self.tokenizer_one(
                            batch,
                            padding="max_length",
                            max_length=self.tokenizer_one.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids
                        
                        # Pin memory and record stream
                        pin_tensor_(batch_tokens_1)
                        tensors_record_stream(stream_1, batch_tokens_1)
                        
                    # Tokenize second encoder  
                    with stream_2:
                        batch_tokens_2 = self.tokenizer_two(
                            batch,
                            padding="max_length",
                            max_length=self.tokenizer_two.model_max_length,
                            truncation=True, 
                            return_tensors="pt"
                        ).input_ids
                        
                        pin_tensor_(batch_tokens_2)
                        tensors_record_stream(stream_2, batch_tokens_2)
                        
                    # Move to device if needed
                    if not device_equals(batch_tokens_1.device, self.device):
                        tensors_to_device_([batch_tokens_1, batch_tokens_2], self.device)
                        
                    tokens_1.append(batch_tokens_1)
                    tokens_2.append(batch_tokens_2)
                    
                    # Clean up pinned memory
                    unpin_tensor_(batch_tokens_1)
                    unpin_tensor_(batch_tokens_2)
                    
            # Combine tokens
            tokens_1 = torch.cat(tokens_1)
            tokens_2 = torch.cat(tokens_2)

            # Encode tokens
            with torch.no_grad():
                # Process encoders with optimized CLIP encoding
                with create_stream_context() as (stream_1, stream_2):
                    with stream_1:
                        text_embeddings_1, _ = encode_clip(
                            text_encoder=self.text_encoder_one,
                            tokens=tokens_1,
                            default_layer=-2,
                            layer_skip=0,
                            use_attention_mask=False,
                            add_layer_norm=False
                        )
                        pin_tensor_(text_embeddings_1)
                        tensors_record_stream(stream_1, text_embeddings_1)
                        
                    with stream_2:
                        text_embeddings_2, pooled_embeddings = encode_clip(
                            text_encoder=self.text_encoder_two,
                            tokens=tokens_2,
                            default_layer=-2,
                            layer_skip=0,
                            add_pooled_output=True,
                            use_attention_mask=False,
                            add_layer_norm=False
                        )
                        pin_tensor_(text_embeddings_2)
                        pin_tensor_(pooled_embeddings)
                        tensors_record_stream(stream_2, [text_embeddings_2, pooled_embeddings])
                        
                    # Combine embeddings
                    prompt_embeds = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
                    
                    # Clean up
                    unpin_tensor_(text_embeddings_1)
                    unpin_tensor_(text_embeddings_2)
                    unpin_tensor_(pooled_embeddings)

            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_embeddings
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
