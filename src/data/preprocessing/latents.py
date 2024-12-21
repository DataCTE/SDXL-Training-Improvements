"""Latent preprocessing utilities for SDXL training."""
import traceback
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto

class LatentPreprocessingError(Exception):
    """Base exception for latent preprocessing errors."""
    pass

class TextEncodingError(LatentPreprocessingError):
    """Raised when text encoding fails."""
    pass

class VAEEncodingError(LatentPreprocessingError):
    """Raised when VAE encoding fails."""
    pass

class CacheError(LatentPreprocessingError):
    """Raised when cache operations fail."""
    pass

class ValidationError(LatentPreprocessingError):
    """Raised when tensor validation fails."""
    pass

class ProcessingStage(Enum):
    """Enum for tracking processing stages."""
    TEXT_ENCODING = auto()
    VAE_ENCODING = auto()
    CACHING = auto()
    VALIDATION = auto()

@dataclass
class ProcessingStats:
    """Statistics for preprocessing operations."""
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    empty_captions: int = 0
    invalid_tensors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
from src.core.logging.logging import setup_logging
import torch
from typing import Dict, List, Optional, Union
import os
import threading
from pathlib import Path

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.config import Config
from src.models.encoders.clip import encode_clip
from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_,
    torch_gc
)
logger = setup_logging(__name__, level="INFO")

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
        use_cache: bool = True
    ):
        """Initialize the latent preprocessor for SDXL training."""
        """Initialize the latent preprocessor for SDXL training.
        
        Args:
            config: Training configuration
            tokenizer_one: First CLIP tokenizer
            tokenizer_two: Second CLIP tokenizer
            text_encoder_one: First CLIP text encoder
            text_encoder_two: Second CLIP text encoder with projection
            vae: VAE model
            device: Target device
            use_cache: Whether to cache embeddings
        """
        self.config = config
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.device = torch.device(device)
        self.use_cache = use_cache and config.global_config.cache.use_cache

        # Set up cache paths if enabled
        if self.use_cache:
            from src.data.utils.paths import convert_windows_path
            self.cache_dir = Path(convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize cache manager
            from .cache_manager import CacheManager
            self.cache_manager = CacheManager(
                cache_dir=self.cache_dir,
                num_proc=config.global_config.cache.num_proc,
                chunk_size=config.global_config.cache.chunk_size,
                compression=config.global_config.cache.compression,
                verify_hashes=config.global_config.cache.verify_hashes
            )
            
            # Create subdirectories for individual files
            self.text_cache_dir = Path(convert_windows_path(self.cache_dir / "text", make_absolute=True))
            self.image_cache_dir = Path(convert_windows_path(self.cache_dir / "image", make_absolute=True))
            self.text_cache_dir.mkdir(parents=True, exist_ok=True)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear cache if configured
            if config.global_config.cache.clear_cache_on_start:
                self.clear_cache()

    def _validate_tensor(self, tensor: torch.Tensor, name: str, expected_dims: int, 
                        expected_shape: Optional[tuple] = None) -> None:
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} is {type(tensor)}, expected torch.Tensor")
        
        if tensor.dim() != expected_dims:
            raise ValidationError(
                f"{name} has {tensor.dim()} dimensions, expected {expected_dims}"
            )
            
        if expected_shape and tensor.shape != expected_shape:
            raise ValidationError(
                f"{name} has shape {tensor.shape}, expected {expected_shape}"
            )
            
        if torch.isnan(tensor).any():
            raise ValidationError(f"{name} contains NaN values")
            
        if torch.isinf(tensor).any():
            raise ValidationError(f"{name} contains infinite values")

    def encode_prompt(
        self,
        prompt_batch: List[str],
        proportion_empty_prompts: float = 0,
        is_train: bool = True,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts to CLIP embeddings.
        
        Args:
            prompt_batch: List of prompts to encode
            proportion_empty_prompts: Proportion of empty prompts to use
            is_train: Whether this is training data
            return_tensors: Whether to return tensors or keep on device
            
        Returns:
            Dict with prompt_embeds and pooled_prompt_embeds
        """
        try:
            # Process prompts with validation
            captions = []
            stats = ProcessingStats()
            stats.total_samples = len(prompt_batch)
            valid_count = 0
            invalid_texts = []
            
            for idx, caption in enumerate(prompt_batch):
                try:
                    if torch.rand(1).item() < proportion_empty_prompts:
                        captions.append("")
                        stats.empty_captions += 1
                        continue
                        
                    if isinstance(caption, str):
                        text = caption.strip()
                        if text:
                            captions.append(text)
                            valid_count += 1
                            stats.successful_samples += 1
                        else:
                            logger.warning(
                                "Empty caption detected",
                                extra={
                                    'index': idx,
                                    'caption': repr(caption),
                                    'function': '__getitem__',
                                    'line_number': traceback.extract_stack()[-1].lineno,
                                    'file_path': __file__
                                }
                            )
                            invalid_texts.append((idx, caption))
                            stats.empty_captions += 1
                            captions.append("")
                    elif isinstance(caption, (list, tuple)):
                        valid_caption = None
                        # Try each caption in the list
                        for cap in caption:
                            if cap is not None:
                                text = str(cap).strip()
                                if text:
                                    valid_caption = text
                                    break
                                    
                        if valid_caption:
                            captions.append(valid_caption)
                            valid_count += 1
                            stats.successful_samples += 1
                        else:
                            logger.warning(f"No valid caption found in list at index {idx}")
                            invalid_texts.append((idx, caption))
                            stats.empty_captions += 1
                            captions.append("")
                    else:
                        raise ValueError(f"Invalid caption type at index {idx}: {type(caption)}")
                        
                except Exception as e:
                    error_details = {
                        'caption_type': type(caption).__name__,
                        'caption_value': repr(caption),
                        'error_type': type(e).__name__,
                        'error_msg': str(e),
                        'traceback': traceback.format_exc()
                    }
                    logger.error(
                        f"Error processing caption at index {idx}:\n"
                        f"Caption type: {error_details['caption_type']}\n"
                        f"Caption value: {error_details['caption_value']}\n"
                        f"Error type: {error_details['error_type']}\n"
                        f"Error message: {error_details['error_msg']}\n"
                        f"Traceback:\n{error_details['traceback']}"
                    )
                    stats.failed_samples += 1
                    captions.append("")
                    
        except Exception as e:
            error_context = {
                'total_samples': stats.total_samples,
                'successful_samples': stats.successful_samples,
                'failed_samples': stats.failed_samples,
                'empty_captions': stats.empty_captions,
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(
                f"Failed to process prompts:\n"
                f"Processing stats:\n"
                f"- Total samples: {error_context['total_samples']}\n"
                f"- Successful: {error_context['successful_samples']}\n"
                f"- Failed: {error_context['failed_samples']}\n"
                f"- Empty: {error_context['empty_captions']}\n"
                f"Error type: {error_context['error_type']}\n"
                f"Error message: {error_context['error_msg']}\n"
                f"Traceback:\n{error_context['traceback']}"
            )
            raise TextEncodingError(
                f"Failed to process prompts: {error_context['error_msg']}\n"
                f"Stats: {stats.successful_samples}/{stats.total_samples} successful"
            ) from e

        # Tokenize prompts
        try:
            tokens_1 = self.tokenizer_one(
                captions,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            tokens_2 = self.tokenizer_two(
                captions,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
        except Exception as e:
            tokenizer_context = {
                'tokenizer_one_max_length': self.tokenizer_one.model_max_length,
                'tokenizer_two_max_length': self.tokenizer_two.model_max_length,
                'num_captions': len(captions),
                'caption_lengths': [len(c) for c in captions],
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(
                f"Failed to tokenize prompts:\n"
                f"Tokenizer context:\n"
                f"- Tokenizer 1 max length: {tokenizer_context['tokenizer_one_max_length']}\n"
                f"- Tokenizer 2 max length: {tokenizer_context['tokenizer_two_max_length']}\n"
                f"- Number of captions: {tokenizer_context['num_captions']}\n"
                f"- Caption lengths: {tokenizer_context['caption_lengths']}\n"
                f"Error type: {tokenizer_context['error_type']}\n"
                f"Error message: {tokenizer_context['error_msg']}\n"
                f"Traceback:\n{tokenizer_context['traceback']}"
            )
            raise TextEncodingError(
                f"Failed to tokenize prompts: {tokenizer_context['error_msg']}\n"
                f"Caption lengths: min={min(tokenizer_context['caption_lengths'])}, "
                f"max={max(tokenizer_context['caption_lengths'])}"
            ) from e

        # Encode with both encoders
        with torch.no_grad():
            # First encoder
            text_encoder_1_output, _ = encode_clip(
                text_encoder=self.text_encoder_one,
                tokens=tokens_1,
                default_layer=-2
            )

            # Second encoder
            text_encoder_2_output, pooled_prompt_embeds = encode_clip(
                text_encoder=self.text_encoder_two,
                tokens=tokens_2,
                default_layer=-2,
                add_pooled_output=True
            )

            # Combine encoder outputs
            prompt_embeds = torch.concat(
                [text_encoder_1_output, text_encoder_2_output],
                dim=-1
            )

        if return_tensors:
            return {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
            }
        
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds
        }

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Encode images to VAE latents with enhanced error handling."""
        stats = ProcessingStats()
        stats.total_samples = len(pixel_values) if isinstance(pixel_values, torch.Tensor) else len(list(pixel_values))
        """Encode images to VAE latents with optimized memory handling."""
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(list(pixel_values))
            
        # Optimize memory format and pin memory
        pixel_values = pixel_values.to(memory_format=torch.channels_last).float()
        if torch.cuda.is_available():
            pixel_values = pixel_values.pin_memory()
            
        latents = []
        for idx in range(0, len(pixel_values), batch_size):
            batch = pixel_values[idx:idx + batch_size]
            with torch.no_grad():
                # Use CUDA streams for pipelined compute and transfer
                if torch.cuda.is_available():
                    compute_stream = torch.cuda.Stream()
                    transfer_stream = torch.cuda.Stream()
                        
                    try:
                        with torch.cuda.stream(compute_stream):
                            batch_latents = self.vae.encode(batch).latent_dist.sample()
                            batch_latents = batch_latents * self.vae.config.scaling_factor
                            pin_tensor_(batch_latents)
                                
                        with torch.cuda.stream(transfer_stream):
                            transfer_stream.wait_stream(compute_stream)
                            tensors_record_stream(transfer_stream, batch_latents)
                            latents.append(batch_latents.cpu())
                            unpin_tensor_(batch_latents)
                            
                    finally:
                        # Ensure streams are properly synchronized and destroyed
                        if compute_stream is not None:
                            compute_stream.synchronize()
                        if transfer_stream is not None:
                            transfer_stream.synchronize()
                            
                        # Clean up tensors
                        if hasattr(batch_latents, 'detach'):
                            batch_latents = batch_latents.detach()
                            
                        # Delete streams explicitly
                        del compute_stream
                        del transfer_stream
                        torch.cuda.empty_cache()
                else:
                    batch_latents = self.vae.encode(batch).latent_dist.sample()
                    batch_latents = batch_latents * self.vae.config.scaling_factor
                    latents.append(batch_latents.cpu())

        latents = torch.cat(latents)
        return {"model_input": latents}

    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 4,  # Reduced default batch size
        cache: bool = True,
        compression: Optional[str] = "zstd",
        max_retries: int = 3,
        max_memory_usage: float = 0.8  # Max fraction of GPU memory to use
    ) -> Dataset:
        # Initialize processing statistics
        stats = ProcessingStats()
        """Preprocess and cache embeddings for a dataset.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for processing
            cache: Whether to use disk caching
            compression: Compression format ("zstd", "gzip", or None)
            
        Returns:
            Dataset with added embeddings
        """
        # Try loading individual cache files first
        if cache and self.use_cache:
            try:
                all_prompt_embeds = []
                all_pooled_embeds = []
                all_vae_latents = []
                cache_hit = True

                for idx in range(len(dataset)):
                    text_cache = self.text_cache_dir / f"{idx}_text.pt"
                    vae_cache = self.image_cache_dir / f"{idx}_vae.pt"
                    
                    if text_cache.exists() and vae_cache.exists():
                        text_data = torch.load(text_cache)
                        vae_data = torch.load(vae_cache)
                        
                        if "prompt_embeds" in text_data and "pooled_prompt_embeds" in text_data:
                            all_prompt_embeds.append(text_data["prompt_embeds"])
                            all_pooled_embeds.append(text_data["pooled_prompt_embeds"])
                            all_vae_latents.append(vae_data["model_input"])
                        else:
                            cache_hit = False
                            break
                    else:
                        cache_hit = False
                        break

                if cache_hit:
                    logger.info("Successfully loaded individual cached embeddings")
                    dataset = dataset.add_column("prompt_embeds", torch.stack(all_prompt_embeds))
                    dataset = dataset.add_column("pooled_prompt_embeds", torch.stack(all_pooled_embeds))
                    dataset = dataset.add_column("model_input", torch.stack(all_vae_latents))
                    return dataset
                else:
                    logger.info("Some cache files missing, recomputing all embeddings")
                    
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}, recomputing...")

        # Process text embeddings
        logger.info("Computing text embeddings...")
        text_embeddings = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            try:
                # Handle slice indexing properly
                batch_indices = list(range(idx, min(idx + batch_size, len(dataset))))
                # Handle dataset indexing with validation
                batch = []
                for i in batch_indices:
                    try:
                        item = dataset[i]
                        if isinstance(item, dict) and "pixel_values" in item:
                            batch.append(item)
                        else:
                            logger.warning(f"Skipping invalid dataset item at index {i}")
                    except Exception as e:
                        logger.error(
                            "Error accessing dataset item",
                            extra={
                                'index': i,
                                'error_type': type(e).__name__,
                                'error_msg': str(e),
                                'function': '_process_item',
                                'line_number': traceback.extract_stack()[-1].lineno,
                                'file_path': __file__,
                                'traceback': traceback.format_exc(),
                                'input_data': repr(dataset[i]) if i < len(dataset) else None,
                                'stack_info': traceback.extract_stack().format(),
                                'process': os.getpid(),
                                'thread': threading.get_ident()
                            }
                        )
                        continue
                
                if not batch:
                    error_msg = f"No valid items found in batch {idx}"
                    logger.error(error_msg)
                    raise ValidationError(
                        error_msg,
                        context={
                            'batch_index': idx,
                            'batch_size': batch_size,
                            'stage': ProcessingStage.VALIDATION.name,
                            'traceback': traceback.format_exc()
                        }
                    )
                batch_texts = []
                valid_count = 0
                
                # Safely extract and validate text from batch
                for item in batch:
                    text_item = item["text"]
                    try:
                        # Handle list/tuple inputs
                        if isinstance(text_item, (list, tuple)):
                            # Take first non-empty caption if multiple
                            valid_captions = []
                            for caption in text_item:
                                try:
                                    if caption is not None:
                                        text = str(caption).strip()
                                        if text:
                                            valid_captions.append(text)
                                except (TypeError, ValueError) as e:
                                    logger.debug(f"Skipping invalid caption: {str(e)}")
                                    continue
                            
                            if valid_captions:
                                batch_texts.append(valid_captions[0])
                            else:
                                logger.warning("No valid captions found in list/tuple")
                                batch_texts.append("")
                        # Handle string inputs
                        elif isinstance(text_item, str):
                            text = text_item.strip()
                            batch_texts.append(text if text else "")
                        # Handle None or other types
                        else:
                            if text_item is not None:
                                try:
                                    text = str(text_item).strip()
                                    batch_texts.append(text if text else "")
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Could not convert to string: {str(e)}")
                                    batch_texts.append("")
                            else:
                                batch_texts.append("")
                    except Exception as e:
                        logger.warning(f"Error processing text item: {str(e)}")
                        batch_texts.append("")
                
                # Count and log details about valid/invalid captions
                valid_count = sum(1 for t in batch_texts if t)
                invalid_texts = [(i, txt) for i, txt in enumerate(batch_texts) if not txt]
                
                if valid_count == 0:
                    error_msg = f"All captions empty or invalid in batch {idx}"
                    logger.error(error_msg)
                    for i, txt in invalid_texts:
                        logger.error(f"  Invalid caption at position {i}, original input: {repr(batch['text'][i])}")
                    raise TextEncodingError(
                        error_msg,
                        context={
                            'batch_index': idx,
                            'invalid_texts': invalid_texts,
                            'stage': ProcessingStage.TEXT_ENCODING.name,
                            'traceback': traceback.format_exc()
                        }
                    )
                    
                logger.info(f"Processing batch {idx} with {valid_count}/{len(batch_texts)} valid captions")
                if invalid_texts:
                    logger.warning(f"Found {len(invalid_texts)} invalid captions in batch {idx}:")
                    for i, txt in invalid_texts:
                        logger.warning(f"  Position {i}, original input: {repr(batch['text'][i])}")
                        
                # Skip batch if no valid captions
                if valid_count == 0:
                    logger.error(f"Skipping batch {idx}: no valid captions")
                    continue
                    
                try:
                    # Check available GPU memory
                    if torch.cuda.is_available():
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        allocated = torch.cuda.memory_allocated()
                        if allocated > max_memory_usage * total_memory:
                            torch_gc()  # Force garbage collection
                            logger.warning("High GPU memory usage, reducing batch size")
                            current_batch = batch_texts[:len(batch_texts)//2]  # Process half batch
                        else:
                            current_batch = batch_texts

                    embeddings = self.encode_prompt(
                        current_batch,
                        proportion_empty_prompts=self.config.data.proportion_empty_prompts
                    )
                    if embeddings is not None and all(t is not None for t in embeddings.values()):
                        # Validate embedding shapes and content
                        prompt_embeds = embeddings["prompt_embeds"]
                        pooled_embeds = embeddings["pooled_prompt_embeds"]
                        
                        if not isinstance(prompt_embeds, torch.Tensor):
                            raise ValueError(f"prompt_embeds is {type(prompt_embeds)}, expected torch.Tensor")
                        if not isinstance(pooled_embeds, torch.Tensor):
                            raise ValueError(f"pooled_prompt_embeds is {type(pooled_embeds)}, expected torch.Tensor")
                            
                        if prompt_embeds.dim() != 3:
                            raise ValueError(f"prompt_embeds has {prompt_embeds.dim()} dimensions, expected 3")
                        if pooled_embeds.dim() != 2:
                            raise ValueError(f"pooled_prompt_embeds has {pooled_embeds.dim()} dimensions, expected 2")
                            
                        text_embeddings.append(embeddings)
                        logger.debug(f"Successfully processed batch {idx} with shapes: "
                                   f"prompt_embeds={prompt_embeds.shape}, "
                                   f"pooled_embeds={pooled_embeds.shape}")
                    else:
                        logger.error(f"Skipping batch {idx}: embeddings validation failed")
                        if embeddings is None:
                            logger.error("  encode_prompt returned None")
                        else:
                            for k, v in embeddings.items():
                                logger.error(f"  {k}: {type(v)} = {v}")
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {idx}: {str(e)}")
                    logger.error(f"Problematic texts: {repr(batch_texts)}")
            except Exception as e:
                logger.error(f"Error processing text batch {idx}: {str(e)}")
                continue

        # Combine text embedding batches with validation
        if not text_embeddings:
            error_msg = "No valid text embeddings were generated."
            logger.error(error_msg)
            logger.error("Processing statistics:")
            logger.error(f"Total samples: {len(dataset)}")
            logger.error(f"Successful samples: {len(text_embeddings) if text_embeddings else 0}")
            logger.error(f"Failed samples: {len(dataset) - (len(text_embeddings) if text_embeddings else 0)}")
            
            # Safely count empty captions
            empty_count = 0
            for item in dataset:
                try:
                    caption = item['text']
                    if isinstance(caption, (list, tuple)):
                        caption = caption[0] if caption else ""
                    if not str(caption).strip():
                        empty_count += 1
                except Exception as e:
                    logger.warning(f"Error checking caption: {str(e)}")
                    empty_count += 1
                    
            logger.error(f"Empty captions: {empty_count}")
            logger.error("Check input captions and previous log messages for details.")
            raise RuntimeError(error_msg)
            
        try:
            # Thorough embedding validation
            valid_embeds = []
            for e in text_embeddings:
                try:
                    if (e is not None and 
                        isinstance(e, dict) and
                        "prompt_embeds" in e and 
                        "pooled_prompt_embeds" in e and
                        isinstance(e["prompt_embeds"], torch.Tensor) and
                        isinstance(e["pooled_prompt_embeds"], torch.Tensor) and
                        e["prompt_embeds"].dim() == 3 and  # [batch, seq_len, hidden_dim]
                        e["pooled_prompt_embeds"].dim() == 2):  # [batch, hidden_dim]
                        
                        # Validate tensor shapes
                        if (e["prompt_embeds"].size(0) > 0 and 
                            e["pooled_prompt_embeds"].size(0) > 0 and
                            e["prompt_embeds"].size(0) == e["pooled_prompt_embeds"].size(0)):
                            valid_embeds.append(e)
                        else:
                            logger.warning(f"Invalid embedding shapes: prompt={e['prompt_embeds'].shape}, pooled={e['pooled_prompt_embeds'].shape}")
                    else:
                        logger.warning("Embedding validation failed: incorrect types or missing keys")
                except Exception as err:
                    logger.error(f"Error validating embedding: {str(err)}")
                    
            if not valid_embeds:
                # Log detailed error statistics
                stats.total_samples = len(dataset)
                stats.successful_samples = len(text_embeddings) if text_embeddings else 0
                stats.failed_samples = stats.total_samples - stats.successful_samples
                stats.empty_captions = sum(1 for c in dataset['text'] if not str(c).strip())

                error_details = (
                    "No valid embeddings found after validation.\n"
                    f"Processing Statistics:\n"
                    f"- Total samples: {stats.total_samples}\n"
                    f"- Successful: {stats.successful_samples}\n" 
                    f"- Failed: {stats.failed_samples}\n"
                    f"- Empty captions: {stats.empty_captions}\n"
                    f"Check previous log messages for detailed error traces."
                )

                if text_embeddings:
                    # Log details about failed embeddings
                    logger.error("All embeddings failed validation:")
                    for i, e in enumerate(text_embeddings):
                        logger.error(f"Embedding {i}:")
                        if isinstance(e, dict):
                            for k, v in e.items():
                                logger.error(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
                        else:
                            logger.error(f"  Unexpected type: {type(e)}")

                logger.error(error_details)
                raise RuntimeError(error_details)
                
            logger.info(f"Validated {len(valid_embeds)}/{len(text_embeddings)} embedding batches")
                
            # Efficiently combine embeddings with tensor replacement
            text_embeddings = {
                "prompt_embeds": torch.cat([e["prompt_embeds"] for e in valid_embeds]),
                "pooled_prompt_embeds": torch.cat([e["pooled_prompt_embeds"] for e in valid_embeds])
            }
            
            # Replace tensors in place when possible
            for embed in valid_embeds:
                if device_equals(embed["prompt_embeds"].device, text_embeddings["prompt_embeds"].device):
                    replace_tensors_(embed["prompt_embeds"], text_embeddings["prompt_embeds"])
                if device_equals(embed["pooled_prompt_embeds"].device, text_embeddings["pooled_prompt_embeds"].device):
                    replace_tensors_(embed["pooled_prompt_embeds"], text_embeddings["pooled_prompt_embeds"])
            
            logger.info(f"Successfully generated embeddings for {len(valid_embeds)} batches")
            
        except Exception as e:
            raise RuntimeError(f"Failed to concatenate text embeddings: {str(e)}")

        # Process VAE latents
        logger.info("Computing VAE latents...")
        vae_latents = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            try:
                # Handle slice indexing properly
                batch_indices = list(range(idx, min(idx + batch_size, len(dataset))))
                # Handle dataset indexing with detailed validation
                batch = []
                for i in batch_indices:
                    try:
                        item = dataset[i]
                        if not isinstance(item, dict):
                            logger.error(
                                "Invalid dataset item type",
                                extra={
                                    'index': i,
                                    'expected_type': 'dict',
                                    'actual_type': type(item).__name__,
                                    'value': repr(item),
                                    'function': '_process_item',
                                    'line_number': traceback.extract_stack()[-1].lineno,
                                    'file_path': __file__,
                                    'traceback': traceback.format_exc()
                                }
                            )
                            continue
                            
                        required_keys = {"pixel_values", "text"}
                        missing_keys = required_keys - set(item.keys())
                        if missing_keys:
                            logger.error(
                                "Missing required dataset item keys",
                                extra={
                                    'index': i,
                                    'missing_keys': list(missing_keys),
                                    'available_keys': list(item.keys()),
                                    'item': repr(item),
                                    'function': '_process_item',
                                    'line_number': traceback.extract_stack()[-1].lineno,
                                    'file_path': __file__,
                                    'traceback': traceback.format_exc()
                                }
                            )
                            continue
                            
                        # Validate pixel values
                        if not isinstance(item["pixel_values"], torch.Tensor):
                            logger.error(
                                "Invalid pixel_values type",
                                extra={
                                    'index': i,
                                    'expected_type': 'torch.Tensor',
                                    'actual_type': type(item['pixel_values']).__name__,
                                    'value': repr(item['pixel_values']),
                                    'shape': getattr(item['pixel_values'], 'shape', None),
                                    'dtype': getattr(item['pixel_values'], 'dtype', None),
                                    'device': getattr(item['pixel_values'], 'device', None),
                                    'function': '_process_item',
                                    'line_number': traceback.extract_stack()[-1].lineno,
                                    'file_path': __file__,
                                    'traceback': traceback.format_exc()
                                }
                            )
                            continue
                            
                        # Validate text
                        if not isinstance(item["text"], (str, list, tuple)):
                            logger.error(
                                "Invalid text type",
                                extra={
                                    'index': i,
                                    'expected_type': 'str, list, or tuple',
                                    'actual_type': type(item['text']).__name__,
                                    'value': repr(item['text']),
                                    'function': '_process_item',
                                    'line_number': traceback.extract_stack()[-1].lineno,
                                    'file_path': __file__,
                                    'traceback': traceback.format_exc()
                                }
                            )
                            continue
                            
                        batch.append(item)
                        logger.debug(f"Successfully validated dataset item {i}")
                        
                    except Exception as e:
                        logger.error(
                            "Error accessing dataset item",
                            extra={
                                'index': i,
                                'error_type': type(e).__name__,
                                'error_msg': str(e),
                                'function': '_process_item',
                                'line_number': traceback.extract_stack()[-1].lineno,
                                'file_path': __file__,
                                'traceback': traceback.format_exc(),
                                'input_data': repr(dataset[i]) if i < len(dataset) else None,
                                'stack_info': traceback.extract_stack().format(),
                                'process': os.getpid(),
                                'thread': threading.get_ident()
                            }
                        )
                        continue
                # Check memory before VAE encoding
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated()
                    if allocated > int(max_memory_usage * total_memory):
                        torch_gc()
                        logger.warning("High GPU memory usage before VAE encoding")
                        
                # Stack and optimize pixel tensors
                batch_pixels = torch.stack([b.get("pixel_values") for b in batch if b.get("pixel_values") is not None])
                pin_tensor_(batch_pixels)
                if torch.cuda.is_available():
                    tensors_record_stream(torch.cuda.current_stream(), batch_pixels)
                
                # Process VAE in smaller chunks if needed  
                chunk_size = batch_size
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated()
                    if allocated > int(0.7 * total_memory):
                        chunk_size = max(1, batch_size // 2)
                
                # Check if device matches before encoding
                if not device_equals(batch_pixels.device, self.device):
                    batch_pixels = batch_pixels.to(self.device)
                    
                latents = self.encode_images(batch_pixels, batch_size=chunk_size)
                
                # Use replace_tensors_ for efficient memory reuse
                if len(vae_latents) > 0:
                    replace_tensors_(vae_latents[-1], latents)
                vae_latents.append(latents)
                
                # Save complete file latents with validation
                if self.cache_manager is not None and "text_embeddings" in locals():
                    for i, latent in enumerate(latents["model_input"]):
                        file_path = dataset[batch_indices[i]].get("file_path")
                        if file_path and i < len(text_embeddings):
                            try:
                                # Prepare data with proper shapes
                                latent_data = {"model_input": latent.unsqueeze(0)}
                                text_data = {
                                    "prompt_embeds": text_embeddings["prompt_embeds"][i].unsqueeze(0),
                                    "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"][i].unsqueeze(0)
                                }
                            
                                # Validate tensors
                                for tensor_dict in [latent_data, text_data]:
                                    for name, tensor in tensor_dict.items():
                                        if not isinstance(tensor, torch.Tensor):
                                            raise ValueError(f"Invalid tensor type for {name}: {type(tensor)}")
                                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                                            raise ValueError(f"Found NaN/Inf in {name} tensor")
                                        
                                # Save with detailed metadata
                                self.cache_manager.save_preprocessed_data(
                                    latent_data=latent_data,
                                    text_embeddings=text_data,
                                    metadata={
                                        "original_file": str(file_path),
                                        "timestamp": time.time(),
                                        "tensor_shapes": {
                                            "latent": tuple(latent.shape),
                                            "prompt": tuple(text_embeddings["prompt_embeds"][i].shape),
                                            "pooled": tuple(text_embeddings["pooled_prompt_embeds"][i].shape)
                                        },
                                        "dtypes": {
                                            "latent": str(latent.dtype),
                                            "prompt": str(text_embeddings["prompt_embeds"][i].dtype),
                                            "pooled": str(text_embeddings["pooled_prompt_embeds"][i].dtype)
                                        }
                                    },
                                    file_path=file_path
                                )
                                logger.debug(f"Successfully cached file {file_path}")
                            except Exception as e:
                                logger.warning(f"Failed to cache file {file_path}: {str(e)}")
                
                # Clean up intermediate tensors
                try:
                    unpin_tensor_(batch_pixels)
                except:
                    pass
                del batch_pixels, latents, embeddings
                torch_gc()
                
            except Exception as e:
                logger.error(
                    "Error processing batch",
                    extra={
                        'batch_index': idx,
                        'error_type': type(e).__name__,
                        'error_msg': str(e),
                        'function': '_process_batch',
                        'line_number': traceback.extract_stack()[-1].lineno,
                        'file_path': __file__,
                        'traceback': traceback.format_exc(),
                        'batch_size': len(batch) if isinstance(batch, (list, tuple)) else 'N/A',
                        'device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
                        'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    }
                )
                continue

        # Load all processed batches from cache
        all_vae_latents = []
        all_text_embeddings = []
        
        if self.cache_manager is not None:
            for chunk_id in sorted([int(k) for k in self.cache_manager.cache_index["chunks"].keys()]):
                try:
                    chunk_path = Path(self.cache_manager.cache_index["chunks"][str(chunk_id)]["path"])
                    if chunk_path.exists():
                        chunk_data = torch.load(chunk_path)
                        batch_data = chunk_data["tensors"][0]  # First item in tensors list
                        
                        all_vae_latents.append(batch_data["vae_latents"]["model_input"])
                        all_text_embeddings.append(batch_data["text_embeddings"])
                except Exception as e:
                    logger.error(f"Error loading chunk {chunk_id}: {str(e)}")
                    continue
                    
        # Combine all batches
        vae_latents = {
            "model_input": torch.cat(all_vae_latents)
        }
        text_embeddings = {
            "prompt_embeds": torch.cat([e["prompt_embeds"] for e in all_text_embeddings]),
            "pooled_prompt_embeds": torch.cat([e["pooled_prompt_embeds"] for e in all_text_embeddings])
        }

        # Cache individual results with compression
        if cache and self.use_cache:
            logger.info("Caching individual embeddings to disk...")
            try:
                for idx in range(len(dataset)):
                    text_cache = self.text_cache_dir / f"{idx}_text.pt"
                    vae_cache = self.image_cache_dir / f"{idx}_vae.pt"
                    
                    # Prepare individual embeddings
                    text_data = {
                        "prompt_embeds": text_embeddings["prompt_embeds"][idx],
                        "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"][idx]
                    }
                    vae_data = {
                        "model_input": vae_latents["model_input"][idx]
                    }
                    
                    # Save with optional compression
                    if compression == "zstd":
                        torch.save(text_data, text_cache, _use_new_zipfile_serialization=True)
                        torch.save(vae_data, vae_cache, _use_new_zipfile_serialization=True)
                    else:
                        torch.save(text_data, text_cache)
                        torch.save(vae_data, vae_cache)
                        
                logger.info(f"Successfully cached individual embeddings to {self.cache_dir}")
                
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
                # Continue even if caching fails

        # Add processed embeddings to dataset
        dataset = dataset.add_column("prompt_embeds", text_embeddings["prompt_embeds"])
        dataset = dataset.add_column("pooled_prompt_embeds", text_embeddings["pooled_prompt_embeds"])
        dataset = dataset.add_column("model_input", vae_latents["model_input"])

        return dataset

    def clear_cache(self):
        """Clear all embedding caches."""
        import shutil
        if self.text_cache_dir.exists():
            shutil.rmtree(self.text_cache_dir)
            self.text_cache_dir.mkdir(parents=True)
        if self.image_cache_dir.exists():
            shutil.rmtree(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True)
        if self.cache_manager is not None:
            self.cache_manager.clear_cache()
