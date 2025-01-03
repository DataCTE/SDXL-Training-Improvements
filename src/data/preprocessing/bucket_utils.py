"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING
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

def compute_bucket_dims(original_size: Tuple[int, int], buckets: List[Tuple[int, int]], config: Config) -> Tuple[int, int]:
    """Find closest bucket for given dimensions."""
    w, h = original_size[0] // 8, original_size[1] // 8
    target_area = w * h
    target_ratio = w / h
    
    # Default to target size from config if no buckets match
    if not buckets:
        logger.warning("No buckets configured, using default target size")
        return tuple(d // 8 for d in config.global_config.image.target_size)
    
    return min(buckets, key=lambda b: (
        abs(b[0] * b[1] - target_area) / target_area + 
        abs((b[0]/b[1]) - target_ratio)
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
            cache_key = cache_manager.get_cache_key(path)
            cached_data = cache_manager.load_tensors(cache_key)
            
            if cached_data and "vae_latents" in cached_data:
                latents = cached_data["vae_latents"]
                dims = (latents.shape[2] * 8, latents.shape[1] * 8)
                bucket = compute_bucket_dims(dims, buckets, config)
            else:
                # For uncached images, use original image dimensions
                img = Image.open(path)
                bucket = compute_bucket_dims(img.size, buckets, config)
            bucket_indices[bucket].append(idx)
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            # Use target size as fallback
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