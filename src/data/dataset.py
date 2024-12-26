import os
import threading
import traceback
import time
import psutil
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager, nullcontext

import torch
import torch.backends.cudnn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
import torch.nn.functional as F
from tqdm.auto import tqdm

# Force speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

import logging
from src.core.memory.tensor import (
    create_stream_context,
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    torch_sync,
    tensors_to_device_,
    device_equals
)
from .utils.paths import convert_windows_path
from .config import Config
from src.data.preprocessing import (
    LatentPreprocessor, TagWeighter, create_tag_weighter,
    create_tag_weighter_with_index, CacheManager, PreprocessingPipeline
)

logger = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_allocated: int = 0
    peak_memory: int = 0

class DatasetError(Exception):
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class ImageLoadError(DatasetError):
    pass

class BucketingError(DatasetError):
    pass

class ProcessingError(DatasetError):
    pass

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with extreme memory handling and 100x speedups."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        preprocessing_pipeline: Optional['PreprocessingPipeline'] = None,
        tag_weighter: Optional[TagWeighter] = None,
        is_train: bool = True,
        enable_memory_tracking: bool = True,
        max_memory_usage: float = 0.8
    ):
        try:
            super().__init__()
            start_time = time.time()
            
            # Basic initialization
            self.stats = DatasetStats()
            self.config = config
            self.image_paths = image_paths
            self.captions = captions
            self.is_train = is_train
            self.enable_memory_tracking = enable_memory_tracking
            self.max_memory_usage = max_memory_usage

            # Initialize preprocessing components
            if preprocessing_pipeline is None:
                # Create cache manager first
                self.cache_manager = CacheManager(
                    cache_dir=Path(convert_windows_path(config.global_config.cache.cache_dir)),
                    num_proc=config.global_config.cache.num_proc,
                    chunk_size=config.global_config.cache.chunk_size,
                    compression=getattr(config.global_config.cache, 'compression', 'zstd'),
                    verify_hashes=config.global_config.cache.verify_hashes,
                    max_memory_usage=max_memory_usage,
                    enable_memory_tracking=enable_memory_tracking
                )
                
                # Create preprocessing pipeline
                self.preprocessing_pipeline = PreprocessingPipeline(
                    config=config,
                    cache_manager=self.cache_manager,
                    is_train=self.is_train,
                    enable_memory_tracking=enable_memory_tracking
                )
            else:
                self.preprocessing_pipeline = preprocessing_pipeline
                self.cache_manager = preprocessing_pipeline.cache_manager

            # Get references to processors
            self.latent_preprocessor = self.preprocessing_pipeline.latent_preprocessor
            self.embedding_processor = self.preprocessing_pipeline.embedding_processor
            
            # Initialize tag weighter if needed
            self.tag_weighter = tag_weighter or self._create_tag_weighter(config, image_paths)

            # Setup image configuration
            self._setup_image_config()

            # Precompute latents if caching is enabled
            if config.global_config.cache.use_cache:
                self._precompute_latents(image_paths, config)

            # Log initialization stats
            logger.info(f"Dataset initialized in {time.time() - start_time:.2f}s", extra={
                'num_images': len(image_paths),
                'num_captions': len(captions),
                'bucket_info': self.preprocessing_pipeline.get_bucket_info(),
                'cache_enabled': config.global_config.cache.use_cache
            })

        except Exception as e:
            logger.error("Failed to initialize dataset", extra={
                'error': str(e),
                'stack_trace': True
            })
            raise

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset with robust error handling."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            bucket_idx = self.bucket_indices[idx]

            # Get cached data
            if self.cache_manager and self.config.global_config.cache.use_cache:
                cached_data = self.cache_manager.get_cached_item(image_path)
                if cached_data:
                    self.stats.cache_hits += 1
                    return self._process_cached_item(cached_data, image_path, caption, bucket_idx)
            
            self.stats.cache_misses += 1
            
            # Process item if not cached
            with torch.cuda.amp.autocast():
                # Process image
                processed_image = self.preprocessing_pipeline._process_image(image_path)
                if processed_image is None:
                    raise ProcessingError(f"Failed to process image {image_path}")

                # Process text
                processed_text = self.latent_preprocessor.encode_prompt([caption])
                if processed_text is None:
                    raise ProcessingError(f"Failed to process caption for {image_path}")

                # Combine results
                result = {
                    "model_input": processed_image["image_latent"],
                    "text": caption,
                    "bucket_idx": bucket_idx,
                    "image_path": image_path,
                    "original_sizes": [processed_image["metadata"]["original_size"]],
                    "crop_top_lefts": [(0, 0)],  # Default if not specified
                    "target_sizes": [processed_image["metadata"].get("bucket_size", (1024, 1024))]
                }

                # Add text embeddings
                if "embeddings" in processed_text:
                    embeddings = processed_text["embeddings"]
                    result.update({
                        "prompt_embeds": embeddings.get("prompt_embeds"),
                        "pooled_prompt_embeds": embeddings.get("pooled_prompt_embeds"),
                        "prompt_embeds_2": embeddings.get("prompt_embeds_2"),
                        "pooled_prompt_embeds_2": embeddings.get("pooled_prompt_embeds_2")
                    })

                # Cache results if enabled
                if self.cache_manager and self.config.global_config.cache.use_cache:
                    self.cache_manager.save_preprocessed_data(
                        image_latent=processed_image["image_latent"],
                        text_latent=processed_text,
                        metadata={
                            **processed_image["metadata"],
                            **processed_text.get("metadata", {})
                        },
                        file_path=image_path
                    )

                return result

        except Exception as e:
            logger.error(f"Error processing dataset item {idx}", extra={
                'error': str(e),
                'image_path': image_path,
                'stack_trace': True
            })
            # Try next item
            return self.__getitem__((idx + 1) % len(self))

    def _process_cached_item(
        self,
        cached_data: Dict[str, Any],
        image_path: str,
        caption: str,
        bucket_idx: int
    ) -> Dict[str, Any]:
        """Process a cached item without tensor validation."""
        try:
            # Get latent data directly
            latent_data = cached_data.get("latent")
            if latent_data is None:
                raise ValueError("No latent data in cache")

            # Get metadata
            metadata = cached_data.get("metadata", {})

            # Build result dictionary
            result = {
                "model_input": latent_data,
                "text": caption,
                "bucket_idx": bucket_idx,
                "image_path": image_path,
                "original_sizes": [metadata.get("original_size", (1024, 1024))],
                "crop_top_lefts": [metadata.get("crop_top_left", (0, 0))],
                "target_sizes": [metadata.get("target_size", (1024, 1024))]
            }

            # Add text embeddings if available
            text_latent = cached_data.get("text_latent", {})
            if "embeddings" in text_latent:
                embeddings = text_latent["embeddings"]
                for key in ["prompt_embeds", "pooled_prompt_embeds", 
                           "prompt_embeds_2", "pooled_prompt_embeds_2"]:
                    if key in embeddings:
                        result[key] = embeddings[key]

            return result

        except Exception as e:
            raise ProcessingError(f"Failed to process cached item: {str(e)}")


    def _setup_image_config(self):
        """Set up image configuration parameters."""
        self.target_size = tuple(map(int, self.config.global_config.image.target_size))
        self.max_size = tuple(map(int, self.config.global_config.image.max_size))
        self.min_size = tuple(map(int, self.config.global_config.image.min_size))
        self.bucket_step = int(self.config.global_config.image.bucket_step)
        self.max_aspect_ratio = float(self.config.global_config.image.max_aspect_ratio)

        # Get buckets and indices
        self.buckets = self.preprocessing_pipeline.get_aspect_buckets(self.config)
        self.bucket_indices = self.preprocessing_pipeline.assign_aspect_buckets(
            image_paths=self.image_paths
        )

    def _create_tag_weighter(self, config: Config, image_paths: List[str]) -> Optional[TagWeighter]:
        """Create tag weighter if enabled in config."""
        if hasattr(config, 'tag_weighting') and config.tag_weighting.enable_tag_weighting:
            return create_tag_weighter(config, image_paths)
        return None

    def _precompute_latents(self, image_paths: List[str], config: Config) -> None:
        """Precompute latents using preprocessing pipeline."""
        if not self.preprocessing_pipeline:
            raise ValueError("Preprocessing pipeline not initialized")
            
        logger.info("Starting latent precomputation...")
        
        try:
            self.preprocessing_pipeline.precompute_latents(
                image_paths=image_paths,
                batch_size=config.training.batch_size,
                proportion_empty_prompts=config.data.proportion_empty_prompts,
                process_latents=True,
                process_text_embeddings=True,
                separate_passes=True
            )
            
        except Exception as e:
            logger.error(f"Error during latent precomputation: {str(e)}", exc_info=True)
            raise

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        try:
            return len(self.image_paths)
        except Exception as e:
            logger.error("Failed to get dataset length", extra={
                'error': str(e),
                'image_paths_type': type(self.image_paths).__name__,
                'has_image_paths': hasattr(self, 'image_paths')
            })
            # Return 0 to indicate empty dataset on error
            return 0

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dict containing batched tensors for SDXL training:
            {
                'pixel_values': torch.Tensor,          # Image tensors [B, C, H, W]
                'prompt_embeds': torch.Tensor,         # Combined text embeddings from both encoders
                'pooled_prompt_embeds': torch.Tensor,  # Pooled embeddings from text encoder 2
                'model_input': torch.Tensor,           # VAE encoded latents
            }
        """
        try:
            # Use the buffer system for efficient memory handling
            buffers = self._create_collate_buffers(
                batch_size=len(batch),
                example_shape=batch[0]['pixel_values'].shape
            )
            
            # Stack pixel values using pinned memory if available
            if 'pixel_values' in buffers:
                # Copy to pinned buffer
                for i, item in enumerate(batch):
                    buffers['pixel_values'][i].copy_(item['pixel_values'])
                # Transfer to device
                pixel_values = buffers['device_storage'].copy_(buffers['pixel_values'])
            else:
                # Fallback to regular stacking
                pixel_values = torch.stack([item['pixel_values'] for item in batch])

            # Get prompt embeddings using the latent preprocessor
            prompt_batch = [item['prompt'] for item in batch]
            encoded_prompts = self.latent_preprocessor.encode_prompt(prompt_batch)
            
            # Encode images to latent space
            model_input = self.latent_preprocessor.encode_images(pixel_values)

            return {
                'pixel_values': pixel_values,
                'prompt_embeds': encoded_prompts['prompt_embeds'],
                'pooled_prompt_embeds': encoded_prompts['pooled_prompt_embeds'],
                'model_input': model_input['model_input']
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to collate batch: {str(e)}")
    

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    preprocessing_pipeline: Optional['PreprocessingPipeline'] = None,
    tag_weighter: Optional['TagWeighter'] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    """Create and initialize dataset instance.
    
    Args:
        config: Configuration object
        image_paths: List of image file paths
        captions: List of corresponding captions
        preprocessing_pipeline: Optional preprocessing pipeline
        tag_weighter: Optional tag weighting system
        enable_memory_tracking: Whether to track memory usage
        max_memory_usage: Maximum memory usage fraction
        
    Returns:
        Initialized dataset instance
    """
    if preprocessing_pipeline and torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(f"Using GPU device 0: {torch.cuda.get_device_name(0)}")
        
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        preprocessing_pipeline=preprocessing_pipeline,
        tag_weighter=tag_weighter,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
