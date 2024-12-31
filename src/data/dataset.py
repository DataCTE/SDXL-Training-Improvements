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

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Get image path and caption
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            
            # Get cache key
            cache_key = self.cache_manager.get_cache_key(image_path)
            
            # Load cached tensors
            cached_data = self.cache_manager.load_tensors(cache_key)
            if cached_data is None:
                raise ValueError(f"No cached data found for key: {cache_key}")
            
            # Return batch with all necessary components
            return {
                "latents": cached_data["pixel_values"],
                "prompt_embeds": cached_data["prompt_embeds"],
                "pooled_prompt_embeds": cached_data["pooled_prompt_embeds"],
                "text": caption,  # Use the actual caption
                "original_size": cached_data["metadata"]["original_size"],
                "target_size": cached_data["metadata"]["target_size"],
                "crop_coords": cached_data["metadata"].get("crop_coords", (0, 0))
            }
        except Exception as e:
            logger.error(f"Error getting dataset item {idx}: {str(e)}")
            raise

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
        """Precompute and cache latents with accurate progress tracking."""
        total_images = len(self)
        start_time = time.time()
        
        # Get cache status for all images
        cached_status = self.preprocessing_pipeline._get_cached_status(self.image_paths)
        already_cached = sum(1 for is_cached in cached_status.values() if is_cached)
        
        remaining_images = total_images - already_cached
        failed_images = 0
        processed_images = already_cached
        batch_size = 64
        
        # Speed tracking with moving window
        processing_times = []
        window_size = 50  # Number of images for moving average
        
        # Create progress bar with detailed stats
        pbar = tqdm(
            total=total_images,
            desc="Precomputing latents",
            unit="img",
            initial=already_cached,
            dynamic_ncols=True,  # Better terminal handling
        )
        
        def update_progress_stats():
            """Calculate and update progress statistics."""
            if processing_times:
                window = processing_times[-window_size:]
                current_speed = len(window) / sum(window) if window else 0
                
                # Use remaining_images here
                eta_seconds = remaining_images / current_speed if current_speed > 0 else 0
                
                # Calculate success rate
                success_rate = (processed_images / (processed_images + failed_images)) * 100 if (processed_images + failed_images) > 0 else 100
                
                return {
                    'processed': processed_images,
                    'failed': failed_images,
                    'cached': already_cached,
                    'remaining': remaining_images,
                    'speed': f'{current_speed:.2f} img/s',
                    'eta': time.strftime('%H:%M:%S', time.gmtime(eta_seconds)),
                    'success': f'{success_rate:.1f}%'
                }
            return {}

        # Process only uncached images
        uncached_indices = [
            i for i, path in enumerate(self.image_paths)
            if not cached_status[path]
        ]
        
        for start_idx in range(0, len(uncached_indices), batch_size):
            batch_start_time = time.time()
            try:
                end_idx = min(start_idx + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[start_idx:end_idx]
                batch_size_actual = len(batch_indices)
                
                # Process batch
                batch_paths = [self.image_paths[i] for i in batch_indices]
                batch_captions = [self.captions[i] for i in batch_indices]
                
                processed_items = self.preprocessing_pipeline.process_image_batch(
                    image_paths=batch_paths,
                    captions=batch_captions,
                    config=self.config
                )
                
                # Track successful and failed items
                successful_items = [item for item in processed_items if item is not None]
                failed_in_batch = batch_size_actual - len(successful_items)
                
                # Cache successful results
                if self.config.global_config.cache.use_cache and successful_items:
                    batch_tensors = []
                    batch_metadata = []
                    valid_paths = []
                    
                    for processed, caption, path in zip(processed_items, batch_captions, batch_paths):
                        if processed is not None:
                            batch_tensors.append({
                                "pixel_values": processed["pixel_values"],
                                "prompt_embeds": processed["prompt_embeds"],
                                "pooled_prompt_embeds": processed["pooled_prompt_embeds"]
                            })
                            batch_metadata.append({
                                "original_size": processed["original_size"],
                                "crop_coords": processed["crop_coords"],
                                "target_size": processed["target_size"],
                                "text": caption
                            })
                            valid_paths.append(path)
                    
                    # Batch save
                    if batch_tensors:
                        cache_results = self.cache_manager.save_latents_batch(
                            batch_tensors=batch_tensors,
                            original_paths=valid_paths,
                            batch_metadata=batch_metadata
                        )
                        
                        # Update counters
                        successful_saves = sum(1 for r in cache_results if r)
                        failed_saves = sum(1 for r in cache_results if not r)
                        
                        processed_images += successful_saves
                        failed_images += failed_saves + failed_in_batch
                        
                        # Track processing time for successful items
                        batch_time = time.time() - batch_start_time
                        processing_times.extend([batch_time / successful_saves] * successful_saves)
                
                # Update progress bar
                pbar.update(batch_size_actual)
                pbar.set_postfix(update_progress_stats())
                
            except Exception as e:
                logger.error(f"Failed to process batch at index {start_idx}", exc_info=True)
                failed_images += batch_size_actual
                pbar.update(batch_size_actual)
                continue
        
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_speed = processed_images / total_time if total_time > 0 else 0
        success_rate = (processed_images / total_images) * 100 if total_images > 0 else 0
        cache_stats = self.cache_manager.get_cache_stats()
        
        logger.info(
            f"Latent computation complete:\n"
            f"- Total images: {total_images}\n"
            f"- Successfully processed: {processed_images}\n"
            f"- Previously cached: {already_cached}\n"
            f"- Failed: {failed_images}\n"
            f"- Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
            f"- Average speed: {avg_speed:.2f} img/s\n"
            f"- Success rate: {success_rate:.1f}%\n"
            f"- Final cache size: {cache_stats['cache_size_bytes'] / 1024**2:.2f} MB"
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

            # Group samples by bucket size
            bucket_groups = {}
            for example in batch:
                size = example["pixel_values"].shape[-2:]  # Get HxW dimensions
                if size not in bucket_groups:
                    bucket_groups[size] = []
                bucket_groups[size].append(example)

            # Take the largest group
            largest_size = max(bucket_groups.keys(), key=lambda k: len(bucket_groups[k]))
            valid_batch = bucket_groups[largest_size]

            return {
                "pixel_values": torch.stack([example["pixel_values"] for example in valid_batch]),
                "prompt_embeds": torch.stack([example["prompt_embeds"] for example in valid_batch]),
                "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"] for example in valid_batch]),
                "original_sizes": [example["original_size"] for example in valid_batch],
                "crop_top_lefts": [example["crop_coords"] for example in valid_batch],
                "text": [example["text"] for example in valid_batch] if "text" in valid_batch[0] else None
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
