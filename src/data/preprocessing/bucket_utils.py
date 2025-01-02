"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def generate_buckets(config: Config) -> List[Tuple[int, int]]:
    """Generate bucket sizes based on config."""
    buckets = []
    min_size = config.global_config.image.min_size
    max_size = config.global_config.image.max_size
    step = config.global_config.image.bucket_step
    
    # Add supported dimensions first
    buckets.extend(tuple(dims) for dims in config.global_config.image.supported_dims)
    
    # Generate additional buckets within bounds
    for h in range(min_size[1], max_size[1] + step, step):
        for w in range(min_size[0], max_size[0] + step, step):
            # Check aspect ratio
            if max(w/h, h/w) <= config.global_config.image.max_aspect_ratio:
                bucket = (w, h)
                if bucket not in buckets:
                    buckets.append(bucket)
                    
    return sorted(buckets)

def compute_bucket_dims(original_size: Tuple[int, int], buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Compute bucket dimensions for an image size."""
    w, h = original_size
    aspect_ratio = w / h
    bucket_ratios = [(bw/bh, (bw,bh)) for bw,bh in buckets]
    _, bucket_dims = min(
        [(abs(aspect_ratio - ratio), dims) for ratio, dims in bucket_ratios]
    )
    return bucket_dims

def validate_aspect_ratio(width: int, height: int, max_ratio: float) -> bool:
    """Validate if the aspect ratio is within acceptable bounds."""
    aspect_ratio = width / height
    return 1/max_ratio <= aspect_ratio <= max_ratio

def group_images_by_bucket(
    image_paths: List[str], 
    cache_manager: "CacheManager"
) -> Dict[Tuple[int, int], List[int]]:
    """Group image indices by their bucket dimensions using cached information."""
    bucket_indices = defaultdict(list)
    
    logger.info("Grouping images into buckets...")
    for idx, image_path in enumerate(tqdm(image_paths, desc="Grouping images")):
        cache_key = cache_manager.get_cache_key(image_path)
        cache_entry = cache_manager.cache_index["entries"].get(cache_key)
        
        if cache_entry and "bucket_dims" in cache_entry:
            bucket_dims = tuple(cache_entry["bucket_dims"])
            bucket_indices[bucket_dims].append(idx)
        else:
            logger.warning(f"Missing bucket dimensions for {image_path}, skipping")
            continue
    
    if not bucket_indices:
        raise RuntimeError("No valid bucket dimensions found in cache. Please preprocess the dataset first.")
    
    return dict(bucket_indices)

def log_bucket_statistics(bucket_indices: Dict[Tuple[int, int], List[int]]):
    """Log statistics about bucket distribution."""
    total_images = sum(len(indices) for indices in bucket_indices.values())
    logger.info(f"\nBucket statistics ({total_images} total images):")
    
    for bucket_dims, indices in sorted(
        bucket_indices.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        w, h = bucket_dims
        logger.info(
            f"Bucket {w}x{h} (ratio {w/h:.2f}): "
            f"{len(indices)} images ({len(indices)/total_images*100:.1f}%)"
        ) 