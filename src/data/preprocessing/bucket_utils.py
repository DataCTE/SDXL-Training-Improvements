"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config
import torch
from PIL import Image
import numpy as np
from src.data.preprocessing.bucket_types import BucketDimensions, BucketInfo

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def generate_buckets(config: Config) -> List[BucketInfo]:
    """Generate comprehensive bucket information with validation."""
    image_config = config.global_config.image
    buckets = []
    
    for idx, dims in enumerate(image_config.supported_dims):
        try:
            w, h = dims[0], dims[1]
            
            # Basic dimension validation
            if not (w >= image_config.min_size[0] and h >= image_config.min_size[1] and
                   w <= image_config.max_size[0] and h <= image_config.max_size[1] and
                   w % image_config.bucket_step == 0 and h % image_config.bucket_step == 0):
                continue
            
            # Create and validate bucket
            bucket = BucketInfo.from_dims(w, h, len(buckets))
            if not bucket.validate():
                logger.warning(f"Invalid bucket configuration: {w}x{h}")
                continue
            
            if validate_aspect_ratio(w, h, image_config.max_aspect_ratio):
                buckets.append(bucket)
                
                # Add flipped dimension if valid
                if h != w:
                    flipped = BucketInfo.from_dims(h, w, len(buckets))
                    if (validate_aspect_ratio(h, w, image_config.max_aspect_ratio) and 
                        flipped.validate()):
                        buckets.append(flipped)
        
        except Exception as e:
            logger.warning(f"Error creating bucket for dims {dims}: {e}")
    
    # Sort buckets by total pixels
    buckets.sort(key=lambda x: x.dimensions.total_pixels)
    
    # Log comprehensive bucket information
    if logger.isEnabledFor(logging.INFO):
        logger.info("\nConfigured buckets:")
        for bucket in buckets:
            dims = bucket.dimensions
            logger.info(
                f"- {dims.width}x{dims.height} pixels "
                f"(latent: {dims.width_latent}x{dims.height_latent}, "
                f"ratio: {dims.aspect_ratio:.2f}, inverse: {dims.aspect_ratio_inverse:.2f}, "
                f"pixels: {dims.total_pixels}, latents: {dims.total_latents}, "
                f"class: {bucket.size_class}/{bucket.aspect_class})"
            )
    
    return buckets

def compute_bucket_dims(
    original_size: Tuple[int, int],
    buckets: List[BucketInfo],
    max_size_diff: float = 0.25  # Maximum allowed size difference
) -> Optional[BucketInfo]:
    """Find closest bucket with improved metrics and validation."""
    w, h = original_size
    original_ratio = w / h
    original_pixels = w * h
    
    # Add size constraints
    valid_buckets = [
        b for b in buckets
        if (abs(b.total_pixels - original_pixels) / original_pixels) <= max_size_diff
    ]
    
    if not valid_buckets:
        return None
        
    # Enhanced metric calculation
    def compute_bucket_score(bucket: BucketInfo) -> float:
        size_diff = abs(bucket.total_pixels - original_pixels) / original_pixels
        ratio_diff = abs(bucket.aspect_ratio - original_ratio)
        
        # Weighted scoring
        return size_diff * 0.6 + ratio_diff * 0.4
    
    return min(valid_buckets, key=compute_bucket_score)

def group_images_by_bucket(
    image_paths: List[str],
    cache_manager: "CacheManager"
) -> Dict[Tuple[int, int], List[int]]:
    """Group images by pixel dimensions with comprehensive caching."""
    bucket_indices = defaultdict(list)
    config = cache_manager.config
    
    # Generate buckets with comprehensive information
    buckets = generate_buckets(config)
    
    if not buckets:
        raise ValueError("No valid buckets generated from config")
    
    # Process images with detailed progress
    for idx, path in enumerate(tqdm(image_paths, desc="Grouping images by bucket")):
        try:
            # Check cache first
            cache_key = cache_manager.get_cache_key(path)
            cached_entry = cache_manager.cache_index["entries"].get(cache_key)
            
            if cached_entry and "bucket_info" in cached_entry:
                # Use cached bucket information
                bucket_info = cached_entry["bucket_info"]
                bucket_indices[tuple(bucket_info["pixel_dims"])].append(idx)
            else:
                # Process new image
                img = Image.open(path)
                bucket_info = compute_bucket_dims(img.size, buckets)
                
                # Store comprehensive bucket information
                bucket_indices[bucket_info.pixel_dims].append(idx)
                
                # Update cache with full bucket information
                if cached_entry:
                    cached_entry["bucket_info"] = {
                        "dimensions": bucket_info.dimensions.__dict__,  # Store all dimension attributes
                        "pixel_dims": bucket_info.pixel_dims,
                        "latent_dims": bucket_info.latent_dims,
                        "bucket_index": bucket_info.bucket_index,
                        "size_class": bucket_info.size_class,
                        "aspect_class": bucket_info.aspect_class
                    }
                    cache_manager._save_index()
            
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            # Use default bucket as fallback
            default_dims = tuple(config.global_config.image.target_size)
            bucket_indices[default_dims].append(idx)
    
    return dict(bucket_indices)

def log_bucket_statistics(bucket_indices: Dict[Tuple[int, int], List[int]], total_images: int):
    """Log comprehensive bucket distribution statistics."""
    logger.info(f"\nBucket statistics ({total_images} total images):")
    
    # Sort buckets by usage
    sorted_buckets = sorted(
        bucket_indices.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # Calculate and log detailed statistics
    for bucket_dims, indices in sorted_buckets:
        w, h = bucket_dims
        count = len(indices)
        percentage = count / total_images * 100
        pixels = w * h
        latents = (w // 8) * (h // 8)
        
        logger.info(
            f"Bucket {w}x{h} "
            f"(ratio: {w/h:.2f}, pixels: {pixels}, latents: {latents}): "
            f"{count} images ({percentage:.1f}%)"
        )

def validate_aspect_ratio(width: int, height: int, max_ratio: float) -> bool:
    """Check if dimensions satisfy max aspect ratio constraint."""
    ratio = width / height
    return 1/max_ratio <= ratio <= max_ratio 