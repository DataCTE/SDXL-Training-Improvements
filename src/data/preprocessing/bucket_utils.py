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

def generate_buckets(config: Config) -> List[Tuple[int, int]]:
    """Generate bucket dimensions from config."""
    image_config = config.global_config.image
    buckets = set()  # Use set to prevent duplicates
    
    # Add supported dimensions from config (converting to latent space dimensions)
    for dims in image_config.supported_dims:
        # Convert pixel dimensions to latent dimensions (divide by 8)
        w, h = dims[0] // 8, dims[1] // 8
        if validate_aspect_ratio(w, h, image_config.max_aspect_ratio):
            buckets.add((w, h))
            # Also add the flipped dimension if it's valid and different
            if h != w and validate_aspect_ratio(h, w, image_config.max_aspect_ratio):
                buckets.add((h, w))
    
    # Convert to sorted list
    buckets = sorted(buckets, key=lambda x: (x[0] * x[1], x[0]))
    
    logger.info("\nConfigured buckets:")
    for w, h in buckets:
        logger.info(f"- {w*8}x{h*8} (ratio {w/h:.2f})")
    
    return buckets

def compute_bucket_dims(original_size: Tuple[int, int], buckets: List[Tuple[int, int]], config: Optional[Config] = None) -> Tuple[int, int]:
    """Find closest bucket for given dimensions with aspect ratio tolerance."""
    w, h = original_size[0] // 8, original_size[1] // 8
    target_area = w * h
    target_ratio = w / h
    
    # Allow 10% tolerance for aspect ratio and area matching
    aspect_tolerance = 0.1
    area_tolerance = 0.1
    
    # Default to target size from config if no buckets match
    if not buckets:
        if config is None:
            raise ValueError("No buckets and no config provided")
        logger.warning("No buckets configured, using default target size")
        return tuple(d // 8 for d in config.global_config.image.target_size)
    
    # Find bucket with closest aspect ratio and area within tolerances
    return min(buckets, key=lambda b: (
        abs(b[0] * b[1] - target_area) / target_area * (1 - area_tolerance) + 
        abs((b[0]/b[1]) - target_ratio) * (1 - aspect_tolerance)
    ))

def group_images_by_bucket(
    image_paths: List[str], 
    cache_manager: "CacheManager"
) -> Dict[Tuple[int, int], List[int]]:
    """Group images by bucket dimensions using cache."""
    bucket_indices = defaultdict(list)
    config = cache_manager.config
    buckets = generate_buckets(config)
    
    if not buckets:
        raise ValueError("No valid buckets generated from config")
    
    for idx, path in enumerate(tqdm(image_paths, desc="Grouping images")):
        try:
            img = Image.open(path)
            # Find nearest bucket dimensions with config
            bucket = compute_bucket_dims(img.size, buckets)  # Remove config parameter as it's only needed for fallback
            # Convert back to pixel space (multiply by 8)
            target_size = (bucket[0] * 8, bucket[1] * 8)
            
            # Store bucket assignment for resizing during VAE encoding
            cache_key = cache_manager.get_cache_key(path)
            if cache_manager.cache_index["entries"].get(cache_key):
                cache_manager.cache_index["entries"][cache_key]["bucket_dims"] = target_size
            
            bucket_indices[bucket].append(idx)
            
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            default_bucket = tuple(d // 8 for d in config.global_config.image.target_size)
            bucket_indices[default_bucket].append(idx)
    
    log_bucket_statistics(bucket_indices, len(image_paths))
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