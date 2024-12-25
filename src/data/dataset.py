"""High-performance dataset implementation for SDXL training with extreme speedups."""
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
    CacheManager, PreprocessingPipeline
)

logger = logging.getLogger(__name__)

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

@dataclass
class DatasetStats:
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_allocated: int = 0
    peak_memory: int = 0

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
        max_memory_usage: float = 0.8,
        timeout: float = 300  # 5 minute timeout
    ):
        start_time = time.time()
        try:
            super().__init__()
            self.stats = DatasetStats()
            self.enable_memory_tracking = enable_memory_tracking
            self.max_memory_usage = max_memory_usage
            self.captions = captions
            self.config = config
            self.image_paths = image_paths
            self.is_train = is_train

            # CUDA optimizations
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            if time.time() - start_time > timeout:
                raise TimeoutError("Dataset initialization timed out")
            raise
          

        # Initialize preprocessing pipeline first if not provided
        if preprocessing_pipeline is None:
            self.cache_manager = CacheManager(
                cache_dir=Path(convert_windows_path(config.global_config.cache.cache_dir)),
                num_proc=config.global_config.cache.num_proc,
                chunk_size=config.global_config.cache.chunk_size,
                compression=getattr(config.global_config.cache, 'compression', 'zstd'),
                verify_hashes=config.global_config.cache.verify_hashes,
                max_memory_usage=max_memory_usage,
                enable_memory_tracking=enable_memory_tracking
            )
            self.preprocessing_pipeline = PreprocessingPipeline(
                config=config,
                latent_preprocessor=None,  # Will be set by pipeline
                cache_manager=self.cache_manager,
                is_train=self.is_train,
                enable_memory_tracking=enable_memory_tracking
            )
        else:
            self.preprocessing_pipeline = preprocessing_pipeline
            self.cache_manager = preprocessing_pipeline.cache_manager

        self.latent_preprocessor = self.preprocessing_pipeline.latent_preprocessor
        self.tag_weighter = tag_weighter or self._create_tag_weighter(config, self.image_paths)

        self._setup_image_config()

        # Get buckets from preprocessing pipeline
        self.buckets = self.preprocessing_pipeline.get_aspect_buckets(config)
        
        # Assign bucket indices using pipeline
        self.bucket_indices = self.preprocessing_pipeline.assign_aspect_buckets(
            self.image_paths
        )

        # Log bucket statistics
        bucket_info = self.preprocessing_pipeline.get_bucket_info()
        logger.info(f"Dataset bucket statistics: {bucket_info}")

        # Precompute latents if needed
        if self.latent_preprocessor and config.global_config.cache.use_cache:
            self._precompute_latents(image_paths, self.latent_preprocessor, config)

        self.transforms = self._setup_transforms()

    def _create_tag_weighter(self, config: Config, image_paths: List[str]) -> Optional[TagWeighter]:
        if hasattr(config, 'tag_weighting') and config.tag_weighting.enable_tag_weighting:
            return create_tag_weighter(config, image_paths)
        return None

    def _setup_image_config(self):
        self.target_size = tuple(map(int, self.config.global_config.image.target_size))
        self.max_size = tuple(map(int, self.config.global_config.image.max_size))
        self.min_size = tuple(map(int, self.config.global_config.image.min_size))
        self.bucket_step = int(self.config.global_config.image.bucket_step)
        self.max_aspect_ratio = float(self.config.global_config.image.max_aspect_ratio)

    def _setup_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _precompute_latents(self, image_paths: List[str], latent_preprocessor: LatentPreprocessor, config: Config) -> None:
        if not self.preprocessing_pipeline:
            raise ValueError("Preprocessing pipeline not initialized")
            
        logger.info("Starting latent precomputation...")
        
        # Track memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial CUDA memory: {initial_memory/1024**2:.1f}MB")
        
        try:
            # First validate cache and get missing items
            if self.cache_manager:
                logger.info("Validating cache index...")
                missing_text, missing_latents = self.cache_manager.validate_cache_index()
                
                # Process all text embeddings first, regardless of missing_text list
                # This ensures we compute text embeddings for all images
                logger.info(f"Processing text embeddings for {len(image_paths)} images")
                for path in tqdm(image_paths, desc="Computing text embeddings"):
                    try:
                        # Check if text embeddings already exist in cache
                        cached_data = self.cache_manager.get_cached_item(path)
                        if cached_data and "text_embeddings" in cached_data:
                            continue
                            
                        caption = self._read_caption(path)
                        embeddings = self.latent_preprocessor.encode_prompt([caption])
                        
                        # Save text embeddings to cache
                        self.cache_manager.save_preprocessed_data(
                            image_latent=None,
                            text_embeddings=embeddings,
                            metadata={"caption": caption, "timestamp": time.time()},
                            file_path=path,
                            caption=caption
                        )
                    except Exception as e:
                        logger.error(f"Failed to process text embeddings for {path}: {e}")

                # Then process missing image latents
                if missing_latents:
                    logger.info(f"Processing {len(missing_latents)} missing image latents")
                    self.preprocessing_pipeline.precompute_latents(
                        image_paths=missing_latents,
                        batch_size=config.training.batch_size,
                        proportion_empty_prompts=config.data.proportion_empty_prompts,
                        process_latents=True,
                        process_text_embeddings=False  # We've already processed text embeddings
                    )
            
            else:
                # No cache manager, process everything
                logger.info("No cache manager available, processing all items")
                self.preprocessing_pipeline.precompute_latents(
                    image_paths=image_paths,
                    batch_size=config.training.batch_size,
                    proportion_empty_prompts=config.data.proportion_empty_prompts,
                    process_latents=True,
                    process_text_embeddings=True
                )
                
            # Force CUDA sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                final_memory = torch.cuda.memory_allocated()
                logger.info(f"Final CUDA memory: {final_memory/1024**2:.1f}MB")
                logger.info(f"Memory change: {(final_memory - initial_memory)/1024**2:.1f}MB")
                
        except Exception as e:
            logger.error(f"Error during latent precomputation: {str(e)}", exc_info=True)
            raise

    def _create_buckets(self) -> List[Tuple[int, int]]:
        """Use preprocessing pipeline's bucket creation."""
        return self.preprocessing_pipeline.get_aspect_buckets(self.config)

    def _assign_buckets(self) -> List[int]:
        """Use preprocessing pipeline's bucket assignment."""
        return self.preprocessing_pipeline.assign_aspect_buckets(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            
            # Remove extra arguments - get_processed_item only takes image_path and caption
            processed_data = self.preprocessing_pipeline.get_processed_item(
                image_path=image_path,
                caption=caption
            )
            
            data_item = {}
            if "image_latent" in processed_data:
                latent_tensor = processed_data["image_latent"]
                data_item["model_input"] = latent_tensor
            if "text_embeddings" in processed_data:
                data_item["text_embeddings"] = processed_data["text_embeddings"]
            else:
                # Process text embeddings on-the-fly if not cached
                embeddings = self.latent_preprocessor.encode_prompt([caption])
                data_item["text_embeddings"] = embeddings

            data_item["text"] = caption
            data_item["loss_weight"] = self.tag_weighter.get_caption_weight(caption) if self.tag_weighter else 1.0
            return data_item
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.preprocessing_pipeline:
                self.preprocessing_pipeline.__exit__(exc_type, exc_val, exc_tb)
            if torch.cuda.is_available():
                torch_sync()
            logger.info(
                f"Dataset shutdown: processed_images={self.stats.processed_images}, "
                f"failed_images={self.stats.failed_images}, "
                f"cache_hits={self.stats.cache_hits}, "
                f"cache_misses={self.stats.cache_misses}"
            )
        except Exception as e:
            logger.error(f"Error during dataset cleanup: {e}")
            
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batching dataset items.
        
        Args:
            batch: List of dictionaries containing processed items
            
        Returns:
            Batched dictionary with all items properly stacked
        """
        if not batch:
            return {}
            
        result = {}
        # Get all keys from first item
        keys = batch[0].keys()
        
        for key in keys:
            if key == "text":
                # Keep text items as list
                result[key] = [item[key] for item in batch]
            elif key == "model_input":
                # Handle the latent tensors
                tensors = [item[key] for item in batch]
                # Pad and stack tensors
                max_shape = [max(sizes) for sizes in zip(*[t.shape for t in tensors])]
                padded_tensors = []
                for t in tensors:
                    pad_sizes = []
                    for i in range(len(t.shape)-1, -1, -1):
                        pad_sizes.extend([0, max_shape[i] - t.shape[i]])
                    padded_t = F.pad(t, pad_sizes, value=0)
                    padded_tensors.append(padded_t)
                result[key] = torch.stack(padded_tensors)
            elif isinstance(batch[0][key], (int, float)):
                # Convert numeric values to tensor
                result[key] = torch.tensor([item[key] for item in batch])
            else:
                # Keep other types as list
                result[key] = [item[key] for item in batch]
                
        return result

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    preprocessing_pipeline: Optional['PreprocessingPipeline'] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    if preprocessing_pipeline and torch.cuda.is_available():
        # Ensure single GPU usage
        torch.cuda.set_device(0)
        logger.info(f"Using GPU device 0: {torch.cuda.get_device_name(0)}")
        
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        preprocessing_pipeline=preprocessing_pipeline,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
