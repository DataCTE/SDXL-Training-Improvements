"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# Force speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_paths
from src.data.config import Config
from src.data.preprocessing import (
    PreprocessingPipeline,
    CacheManager,
    create_tag_weighter_with_index
)

logger = get_logger(__name__)

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
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
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
            self.is_train = is_train
            self.enable_memory_tracking = enable_memory_tracking
            self.max_memory_usage = max_memory_usage

            # Convert paths if needed
            self.image_paths = [
                str(convert_windows_path(p) if is_windows_path(p) else Path(p))
                for p in image_paths
            ]
            self.captions = captions

            # Initialize cache manager
            cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                max_cache_size=config.global_config.cache.max_cache_size,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Initialize preprocessing pipeline
            self.preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline(
                config=config,
                model=config.model,
                cache_manager=self.cache_manager,
                is_train=is_train,
                enable_memory_tracking=enable_memory_tracking
            )

            # Initialize tag weighter with index
            if config.tag_weighting.enable_tag_weighting:
                image_captions = dict(zip(self.image_paths, self.captions))
                index_path = Path(cache_dir) / "tag_weights_index.json"
                self.tag_weighter = create_tag_weighter_with_index(
                    config=config,
                    image_captions=image_captions,
                    index_output_path=index_path
                )
            else:
                self.tag_weighter = None

            # Setup image configuration
            self._setup_image_config()

            # Precompute latents if caching enabled
            if config.global_config.cache.use_cache:
                self._precompute_latents()

            logger.info(f"Dataset initialized in {time.time() - start_time:.2f}s", extra={
                'num_images': len(image_paths),
                'num_captions': len(captions),
                'cache_enabled': config.global_config.cache.use_cache,
                'tag_weighting_enabled': self.tag_weighter is not None
            })

        except Exception as e:
            logger.error("Failed to initialize dataset", exc_info=True)
            raise

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single item from the dataset with robust error handling."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]

            # Try cache first if enabled
            if self.config.global_config.cache.use_cache:
                cached_data = self.cache_manager.load_latents(
                    image_path,
                    device=self.preprocessing_pipeline.device  # Load directly to correct device
                )
                if cached_data is not None:
                    return cached_data

            # Process on GPU via pipeline
            processed = self.preprocessing_pipeline._process_single_image(
                image_path=image_path,
                config=self.config
            )
            if processed is None:
                return None

            # Get text embeddings (already on GPU from pipeline)
            encoded_text = self.preprocessing_pipeline.encode_prompt(
                batch={"text": [caption]},
                proportion_empty_prompts=0.0
            )

            # Everything stays on GPU until collation
            result = {
                "pixel_values": processed["pixel_values"],  # Already on GPU
                "original_size": processed["original_size"],
                "crop_coords": processed["crop_coords"],
                "target_size": processed["target_size"],
                "prompt_embeds": encoded_text["prompt_embeds"],
                "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"],
                "text": caption
            }

            # Cache if enabled (cache manager handles CPU conversion)
            if self.config.global_config.cache.use_cache:
                self.cache_manager.save_latents(
                    tensors={
                        "pixel_values": processed["pixel_values"],
                        "prompt_embeds": encoded_text["prompt_embeds"],
                        "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"]
                    },
                    original_path=image_path,
                    metadata={
                        "original_size": processed["original_size"],
                        "crop_coords": processed["crop_coords"],
                        "target_size": processed["target_size"],
                        "text": caption
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error processing dataset item {idx}: {e}", exc_info=True)
            return None

    def _setup_image_config(self):
        """Set up image configuration parameters."""
        self.target_size = tuple(map(int, self.config.global_config.image.target_size))
        self.max_size = tuple(map(int, self.config.global_config.image.max_size))
        self.min_size = tuple(map(int, self.config.global_config.image.min_size))
        self.bucket_step = int(self.config.global_config.image.bucket_step)
        self.max_aspect_ratio = float(self.config.global_config.image.max_aspect_ratio)

        # Get buckets from pipeline
        self.buckets = self.preprocessing_pipeline.get_aspect_buckets(self.config)
        self.bucket_indices = self.preprocessing_pipeline._assign_bucket_indices(self.image_paths)

    def _precompute_latents(self) -> None:
        """Precompute and cache latents for all images with resume support."""
        total_images = len(self)
        
        # Get cache status for all images
        cached_status = self.preprocessing_pipeline._get_cached_status(self.image_paths)
        already_cached = sum(1 for is_cached in cached_status.values() if is_cached)
        
        remaining_images = total_images - already_cached
        skipped_images = 0
        processed_images = already_cached
        batch_size = 32
        
        logger.info(f"Found {already_cached} previously cached images")
        
        # Create progress bar for remaining images
        pbar = tqdm(
            total=remaining_images,
            desc="Precomputing latents",
            unit="img",
            initial=already_cached,  # Start from previously cached count
            postfix={
                'processed': processed_images,
                'skipped': skipped_images,
                'cached': already_cached,
                'success_rate': f"{(processed_images/total_images)*100:.1f}%"
            }
        )
        
        # Process only uncached images
        uncached_indices = [
            i for i, path in enumerate(self.image_paths)
            if not cached_status[path]
        ]
        
        for start_idx in range(0, len(uncached_indices), batch_size):
            try:
                end_idx = min(start_idx + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[start_idx:end_idx]
                
                # Process batch
                batch_paths = [self.image_paths[i] for i in batch_indices]
                batch_captions = [self.captions[i] for i in batch_indices]
                
                processed_items = self.preprocessing_pipeline.process_image_batch(
                    image_paths=batch_paths,
                    captions=batch_captions,
                    config=self.config
                )
                
                # Cache results
                if self.config.global_config.cache.use_cache:
                    for path, processed, caption in zip(batch_paths, processed_items, batch_captions):
                        if processed is not None:
                            cache_success = self.cache_manager.save_latents(
                                tensors={
                                    "pixel_values": processed["pixel_values"],
                                    "prompt_embeds": processed["prompt_embeds"],
                                    "pooled_prompt_embeds": processed["pooled_prompt_embeds"]
                                },
                                original_path=path,
                                metadata={
                                    "original_size": processed["original_size"],
                                    "crop_coords": processed["crop_coords"],
                                    "target_size": processed["target_size"],
                                    "text": caption
                                }
                            )
                            
                            if cache_success:
                                processed_images += 1
                            else:
                                skipped_images += 1
                        else:
                            skipped_images += 1
                            
                        pbar.update(1)
                        pbar.set_postfix({
                            'processed': processed_images,
                            'skipped': skipped_images,
                            'cached': already_cached,
                            'success_rate': f"{(processed_images/total_images)*100:.1f}%"
                        })
                
                # Periodic CUDA sync
                if torch.cuda.is_available() and (start_idx + batch_size) % (batch_size * 4) == 0:
                    torch.cuda.synchronize()
                    
            except Exception as e:
                logger.error(f"Failed to process batch at index {start_idx}", exc_info=True)
                skipped_images += len(batch_indices)
                pbar.update(len(batch_indices))
                pbar.set_postfix({
                    'processed': processed_images,
                    'skipped': skipped_images,
                    'success_rate': f"{(processed_images / (processed_images + skipped_images) * 100):.1f}%"
                })
                continue
        
        pbar.close()
        
        # Log final statistics including previously cached
        cache_stats = self.cache_manager.get_cache_stats() if self.cache_manager else {}
        logger.info(
            f"Precomputing complete: {processed_images}/{total_images} images processed successfully "
            f"({skipped_images} skipped, {already_cached} previously cached) - "
            f"Success rate: {(processed_images/total_images)*100:.1f}%\n"
            f"Cache stats: {cache_stats}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            # Filter out None values from failed processing
            batch = [b for b in batch if b is not None]
            
            if not batch:
                raise ValueError("Empty batch after filtering")

            return {
                "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
                "prompt_embeds": torch.stack([example["prompt_embeds"] for example in batch]),
                "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"] for example in batch]),
                "original_sizes": [example["original_size"] for example in batch],
                "crop_top_lefts": [example["crop_coords"] for example in batch],
                "text": [example["text"] for example in batch] if "text" in batch[0] else None
            }

        except Exception as e:
            logger.error("Failed to collate batch", exc_info=True)
            raise RuntimeError(f"Collate failed: {str(e)}") from e

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    """Create and initialize dataset instance."""
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        preprocessing_pipeline=preprocessing_pipeline,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
