import logging
import traceback
import time
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, Union
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
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
        self._setup_cache(config, use_cache)
        self.config = config
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one 
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        self.device = device
        
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
                # Process encoders in parallel streams  
                with create_stream_context() as (stream_1, stream_2):
                    with stream_1:
                        text_embeddings_1 = self.text_encoder_one(tokens_1)[0]
                        pin_tensor_(text_embeddings_1)
                        tensors_record_stream(stream_1, text_embeddings_1)
                        
                    with stream_2:
                        text_embeddings_2, pooled_embeddings = self.text_encoder_two(tokens_2, output_hidden_states=True)
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
                            
                        with torch.no_grad():
                            batch_latents = self.vae.encode(batch).latent_dist.sample()
                            batch_latents = batch_latents * self.vae.config.scaling_factor
                            
                        # Pin and record stream
                        pin_tensor_(batch_latents)
                        tensors_record_stream(compute_stream, batch_latents)
                        
                        # Store result
                        latents.append(batch_latents.cpu())
                        
                        # Clean up
                        unpin_tensor_(batch_latents)
                        
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
