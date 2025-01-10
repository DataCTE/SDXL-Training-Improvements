"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import logging
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
    """Generate comprehensive bucket information with dynamic sizing."""
    logger.info("Generating training buckets", extra={
        'min_size': config.global_config.image.min_size,
        'max_size': config.global_config.image.max_size,
        'bucket_step': config.global_config.image.bucket_step
    })
    
    buckets = []
    min_w, min_h = config.global_config.image.min_size
    max_w, max_h = config.global_config.image.max_size
    step = 64  # More granular step size
    
    # Generate base dimensions
    widths = list(range(min_w, max_w + 1, step))
    heights = list(range(min_h, max_h + 1, step))
    
    # Add common resolutions
    common_sizes = [
        (1024, 1024),  # Square
        (1024, 1536), (1536, 1024),  # 2:3 and 3:2
        (1024, 1280), (1280, 1024),  # 4:5 and 5:4
        (1152, 896), (896, 1152),    # Similar to 4:3
        (1216, 832), (832, 1216),    # Similar to 3:2
        (1152, 1152),                # Larger square
        (1280, 1536), (1536, 1280),  # Larger 4:5
        (1408, 1024), (1024, 1408),  # Custom common size
    ]
    
    # Add buckets for common sizes first
    for w, h in common_sizes:
        try:
            bucket = BucketInfo.from_dims(w, h, len(buckets))
            valid, error = validate_bucket_config(bucket, config)
            
            if valid:
                buckets.append(bucket)
                logger.debug(f"Added common bucket {w}x{h}")
            else:
                logger.debug(f"Invalid bucket {w}x{h}: {error}")
        except Exception as e:
            logger.debug(f"Skipping common size {w}x{h}: {e}")
    
    # Generate additional buckets with aspect ratio constraints
    max_ratio = config.global_config.image.max_aspect_ratio
    
    for w in widths:
        for h in heights:
            # Skip if dimensions match existing bucket
            if any(b.pixel_dims == (w, h) for b in buckets):
                continue
                
            try:
                # Check aspect ratio
                ratio = w / h
                if not (1/max_ratio <= ratio <= max_ratio):
                    continue
                
                bucket = BucketInfo.from_dims(w, h, len(buckets))
                valid, error = validate_bucket_config(bucket, config)
                
                if valid:
                    buckets.append(bucket)
                    logger.debug(f"Added dynamic bucket {w}x{h}")
                    
            except Exception as e:
                logger.debug(f"Skipping dimensions {w}x{h}: {e}")
    
    # Sort buckets by total pixels and deduplicate
    buckets.sort(key=lambda x: (x.dimensions.total_pixels, x.dimensions.aspect_ratio))
    
    # Remove buckets that are too similar
    filtered_buckets = []
    for bucket in buckets:
        # Check if this bucket is too similar to any existing filtered bucket
        is_unique = True
        for existing in filtered_buckets:
            size_diff = abs(bucket.dimensions.total_pixels - existing.dimensions.total_pixels) / bucket.dimensions.total_pixels
            aspect_diff = abs(bucket.dimensions.aspect_ratio - existing.dimensions.aspect_ratio)
            
            if size_diff < 0.1 and aspect_diff < 0.1:  # 10% similarity threshold
                is_unique = False
                break
                
        if is_unique:
            filtered_buckets.append(bucket)
    
    # Log final bucket statistics
    log_bucket_statistics(
        {bucket.pixel_dims: [i] for i, bucket in enumerate(filtered_buckets)},
        len(filtered_buckets)
    )
    
    return filtered_buckets

def compute_bucket_dims(
    original_size: Tuple[int, int],
    buckets: List[BucketInfo],
    max_size_diff: float = 0.3,
    max_aspect_diff: float = 0.15
) -> Optional[BucketInfo]:
    """Find closest bucket with improved metrics and validation."""
    try:
        if not buckets:
            logger.warning("No buckets available")
            return None
            
        w, h = original_size
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid image dimensions: {w}x{h}")
            return None
            
        original_ratio = w / h
        original_pixels = w * h
        
        # Find best matching bucket with weighted scoring
        best_bucket = None
        best_score = float('inf')
        
        for bucket in buckets:
            try:
                # Calculate size difference as percentage
                size_diff = abs(bucket.dimensions.total_pixels - original_pixels) / original_pixels
                
                # Calculate aspect ratio difference with tolerance for similar ratios
                aspect_diff = abs(bucket.dimensions.aspect_ratio - original_ratio)
                if aspect_diff > 1:  # Handle reciprocal aspect ratios
                    aspect_diff = abs(1/bucket.dimensions.aspect_ratio - original_ratio)
                
                # Weight size difference more heavily for extreme cases
                size_weight = 0.7 if size_diff > 0.2 else 0.5
                aspect_weight = 1.0 - size_weight
                
                if size_diff <= max_size_diff and aspect_diff <= max_aspect_diff:
                    score = size_diff * size_weight + aspect_diff * aspect_weight
                    if score < best_score:
                        best_score = score
                        best_bucket = bucket
            except Exception as e:
                logger.debug(f"Error evaluating bucket: {e}")
                continue
        
        if best_bucket is None:
            # Create a new bucket if none found
            try:
                # Round dimensions to nearest bucket step
                step = 64  # More granular bucket step
                w = ((w + step - 1) // step) * step
                h = ((h + step - 1) // step) * step
                
                # Create new bucket
                new_bucket = BucketInfo.from_dims(w, h, len(buckets))
                logger.info(f"Created new bucket for size {w}x{h}")
                return new_bucket
            except Exception as e:
                logger.error(f"Failed to create new bucket: {e}")
                return None
        
        return best_bucket
        
    except Exception as e:
        logger.error(f"Failed to compute bucket dimensions: {e}")
        return None

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
    """Validate bucket configuration with flexible constraints."""
    try:
        image_config = config.global_config.image
        w, h = bucket.pixel_dims
        
        # More flexible validation checks
        min_w, min_h = image_config.min_size
        max_w, max_h = image_config.max_size
        
        # Allow some tolerance around min/max sizes
        size_tolerance = 0.1  # 10% tolerance
        min_w = int(min_w * (1 - size_tolerance))
        min_h = int(min_h * (1 - size_tolerance))
        max_w = int(max_w * (1 + size_tolerance))
        max_h = int(max_h * (1 + size_tolerance))
        
        # First validate internal bucket consistency
        valid, error = bucket.validate_with_details()
        if not valid:
            return False, f"Internal validation failed: {error}"
        
        # Then validate against config constraints
        checks = [
            (min_w <= w <= max_w,
             f"Width {w} outside allowed range {min_w}-{max_w}"),
            
            (min_h <= h <= max_h,
             f"Height {h} outside allowed range {min_h}-{max_h}"),
            
            # More flexible step size validation
            (w % 8 == 0 and h % 8 == 0,
             f"Dimensions must be divisible by 8: {w}x{h}"),
            
            # More flexible aspect ratio validation
            (validate_aspect_ratio(w, h, image_config.max_aspect_ratio * 1.2),  # 20% more tolerance
             f"Aspect ratio {w/h:.2f} outside allowed range")
        ]
        
        # Check each validation condition
        for condition, error_message in checks:
            if not condition:
                return False, error_message
            
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}" 
