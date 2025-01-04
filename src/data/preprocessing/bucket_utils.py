"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config
import torch
from PIL import Image

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def generate_buckets(config: Config) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Generate bucket dimensions from config, returning both pixel and latent dimensions."""
    image_config = config.global_config.image
    buckets = []  # [(pixel_dims, latent_dims), ...]
    
    # Work in pixel space
    for dims in image_config.supported_dims:
        w, h = dims[0], dims[1]  # Pixel dimensions
        
        # Validate dimensions against min/max size and step
        if (w >= image_config.min_size[0] and h >= image_config.min_size[1] and
            w <= image_config.max_size[0] and h <= image_config.max_size[1] and
            w % image_config.bucket_step == 0 and h % image_config.bucket_step == 0):
            
            if validate_aspect_ratio(w, h, image_config.max_aspect_ratio):
                # Store both pixel and latent dimensions
                w_latent, h_latent = w // 8, h // 8
                buckets.append(((w, h), (w_latent, h_latent)))
                
                # Also add the flipped dimension if valid
                if h != w and validate_aspect_ratio(h, w, image_config.max_aspect_ratio):
                    flipped = ((h, w), (h_latent, w_latent))
                    if flipped not in buckets:
                        buckets.append(flipped)
    
    # Log both pixel and latent dimensions
    if logger.isEnabledFor(logging.INFO):
        logger.info("\nConfigured buckets:")
        for pixel_dims, latent_dims in buckets:
            w, h = pixel_dims
            w_latent, h_latent = latent_dims
            logger.info(
                f"- {w}x{h} pixels (latent: {w_latent}x{h_latent}, "
                f"ratio: {w/h:.2f})"
            )
    
    return buckets

def compute_bucket_dims(
    original_size: Tuple[int, int], 
    buckets: List[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find closest bucket returning both pixel and latent dimensions."""
    w, h = original_size[0], original_size[1]
    
    # Find closest bucket using pixel dimensions
    closest_bucket_idx = min(
        range(len(buckets)),
        key=lambda i: (
            abs(buckets[i][0][0] * buckets[i][0][1] - (w * h)) / (w * h) +
            abs((buckets[i][0][0]/buckets[i][0][1]) - (w/h))
        )
    )
    
    # Return both pixel and latent dimensions
    return buckets[closest_bucket_idx]

def group_images_by_bucket(
    image_paths: List[str], 
    cache_manager: "CacheManager"
) -> Dict[Tuple[int, int], List[int]]:
    """Group images by pixel dimensions using cache."""
    bucket_indices = defaultdict(list)
    config = cache_manager.config
    
    # Generate buckets with both dimensions
    buckets = generate_buckets(config)
    
    if not buckets:
        raise ValueError("No valid buckets generated from config")
    
    logger.setLevel(logging.WARNING)
    
    for idx, path in enumerate(tqdm(image_paths, desc="Grouping images")):
        try:
            # Check cache
            cache_key = cache_manager.get_cache_key(path)
            cached_entry = cache_manager.cache_index["entries"].get(cache_key)
            
            if cached_entry and "bucket_dims" in cached_entry:
                # Use cached pixel dimensions
                bucket = cached_entry["bucket_dims"]  # Already in pixel space
            else:
                # Compute new bucket dimensions
                img = Image.open(path)
                bucket, _ = compute_bucket_dims(img.size, buckets)  # Use pixel dims
            
            bucket_indices[bucket].append(idx)
            
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            default_bucket = tuple(config.global_config.image.target_size)
            bucket_indices[default_bucket].append(idx)
    
    logger.setLevel(logging.INFO)
    return dict(bucket_indices)

def log_bucket_statistics(bucket_indices: Dict[Tuple[int, int], List[int]], total_images: int):
    """Log bucket distribution statistics."""
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

def validate_aspect_ratio(width: int, height: int, max_ratio: float) -> bool:
    """Check if dimensions satisfy max aspect ratio constraint."""
    ratio = width / height
    return 1/max_ratio <= ratio <= max_ratio 