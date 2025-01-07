"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
from collections import defaultdict
from src.core.logging import UnifiedLogger, LogConfig, ProgressPredictor, get_logger
from src.data.config import Config
import torch
from PIL import Image
import numpy as np
from src.data.preprocessing.bucket_types import BucketDimensions, BucketInfo

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = get_logger(__name__)

def generate_buckets(config: Config) -> List[BucketInfo]:
    """Generate comprehensive bucket information with enhanced validation."""
    logger.info("Generating training buckets", extra={
        'min_size': config.global_config.image.min_size,
        'max_size': config.global_config.image.max_size,
        'bucket_step': config.global_config.image.bucket_step
    })
    
    buckets = []
    for idx, dims in enumerate(config.global_config.image.supported_dims):
        try:
            w, h = dims[0], dims[1]
            bucket = BucketInfo.from_dims(w, h, len(buckets))
            valid, error = validate_bucket_config(bucket, config)
            
            if not valid:
                logger.warning("Invalid bucket configuration", extra={
                    'width': w,
                    'height': h,
                    'error': error
                })
                continue
                
            buckets.append(bucket)
            logger.debug("Added bucket", extra={
                'dims': f"{w}x{h}",
                'aspect_ratio': f"{w/h:.2f}",
                'total_pixels': w*h
            })
            
            # Add flipped dimension if valid
            if h != w:
                flipped = BucketInfo.from_dims(h, w, len(buckets))
                flipped_valid, flipped_error = validate_bucket_config(flipped, config)
                
                if flipped_valid:
                    buckets.append(flipped)
                else:
                    logger.debug(f"Skipping flipped bucket {h}x{w}: {flipped_error}")
        
        except Exception as e:
            logger.warning(f"Error creating bucket for dims {dims}: {e}")
            continue
    
    # Sort buckets by total pixels
    buckets.sort(key=lambda x: x.dimensions.total_pixels)
    
    # Log comprehensive bucket information
    if logger.isEnabledFor(logging.INFO):
        log_bucket_statistics(
            {bucket.pixel_dims: [i] for i, bucket in enumerate(buckets)},
            len(buckets)
        )
    
    return buckets

def compute_bucket_dims(
    original_size: Tuple[int, int],
    buckets: List[BucketInfo],
    max_size_diff: float = 0.25,
    max_aspect_diff: float = 0.1
) -> Optional[BucketInfo]:
    """Find closest bucket with improved metrics and validation.
    
    Args:
        original_size: Original image dimensions (width, height)
        buckets: List of available buckets
        max_size_diff: Maximum allowed size difference ratio (default: 0.25)
        max_aspect_diff: Maximum allowed aspect ratio difference (default: 0.1)
        
    Returns:
        BucketInfo or None if no suitable bucket found
    """
    w, h = original_size
    original_ratio = w / h
    original_pixels = w * h
    
    # Filter valid buckets based on constraints
    valid_buckets = []
    for bucket in buckets:
        size_diff = abs(bucket.dimensions.total_pixels - original_pixels) / original_pixels
        aspect_diff = abs(bucket.dimensions.aspect_ratio - original_ratio)
        
        if size_diff <= max_size_diff and aspect_diff <= max_aspect_diff:
            valid_buckets.append((bucket, size_diff, aspect_diff))
    
    if not valid_buckets:
        return None
    
    # Score buckets using weighted metrics
    def compute_score(size_diff: float, aspect_diff: float) -> float:
        size_weight = 0.6
        aspect_weight = 0.4
        return size_diff * size_weight + aspect_diff * aspect_weight
    
    # Return bucket with lowest score
    return min(valid_buckets, key=lambda x: compute_score(x[1], x[2]))[0]

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
    predictor = ProgressPredictor()
    predictor.start(len(image_paths))
        
    for idx, path in enumerate(image_paths):
        timing = predictor.update(1)
        if idx % 100 == 0:  # Log progress periodically
            eta_str = predictor.format_time(timing["eta_seconds"])
            logger.info(f"Processing images: {idx}/{len(image_paths)} (ETA: {eta_str})")
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

def validate_bucket_config(
    bucket: BucketInfo,
    config: "Config"
) -> Tuple[bool, Optional[str]]:
    """Validate bucket configuration against global settings with enhanced error reporting."""
    try:
        image_config = config.global_config.image
        w, h = bucket.pixel_dims
        
        # Comprehensive validation checks with detailed messages
        checks = [
            (image_config.min_size[0] <= w <= image_config.max_size[0],
             f"Width {w} outside allowed range {image_config.min_size[0]}-{image_config.max_size[0]}"),
            
            (image_config.min_size[1] <= h <= image_config.max_size[1],
             f"Height {h} outside allowed range {image_config.min_size[1]}-{image_config.max_size[1]}"),
            
            (w % image_config.bucket_step == 0,
             f"Width {w} not divisible by bucket_step {image_config.bucket_step}"),
            
            (h % image_config.bucket_step == 0,
             f"Height {h} not divisible by bucket_step {image_config.bucket_step}"),
            
            (validate_aspect_ratio(w, h, image_config.max_aspect_ratio),
             f"Aspect ratio {w/h:.2f} outside allowed range")
        ]
        
        # Check each validation condition
        for condition, error_message in checks:
            if not condition:
                return False, error_message
        
        # Validate internal bucket consistency
        valid, error = bucket.validate_with_details()
        if not valid:
            return False, f"Internal bucket validation failed: {error}"
            
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}" 
