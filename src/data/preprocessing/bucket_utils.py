"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional, NamedTuple
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config
import torch
from PIL import Image
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class BucketDimensions:
    """Explicit storage of all dimension-related information."""
    width: int                       # Pixel width
    height: int                      # Pixel height
    width_latent: int               # Latent width (w//8)
    height_latent: int              # Latent height (h//8)
    aspect_ratio: float             # Width/height ratio
    aspect_ratio_inverse: float     # Height/width ratio
    total_pixels: int               # Total pixel count
    total_latents: int              # Total latent count
    
    @classmethod
    def from_pixels(cls, width: int, height: int) -> 'BucketDimensions':
        """Create dimensions from pixel values with validation."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        
        return cls(
            width=width,
            height=height,
            width_latent=width // 8,
            height_latent=height // 8,
            aspect_ratio=width / height,
            aspect_ratio_inverse=height / width,
            total_pixels=width * height,
            total_latents=(width // 8) * (height // 8)
        )
    
    def validate(self) -> bool:
        """Validate internal consistency of dimensions."""
        checks = [
            self.width > 0,
            self.height > 0,
            self.width_latent == self.width // 8,
            self.height_latent == self.height // 8,
            np.isclose(self.aspect_ratio, self.width / self.height),
            np.isclose(self.aspect_ratio_inverse, 1 / self.aspect_ratio),
            self.total_pixels == self.width * self.height,
            self.total_latents == self.width_latent * self.height_latent
        ]
        return all(checks)

@dataclass
class BucketInfo:
    """Comprehensive bucket information with redundant storage."""
    dimensions: BucketDimensions    # All dimension-related information
    pixel_dims: Tuple[int, int]    # Redundant pixel dimensions (w, h)
    latent_dims: Tuple[int, int]   # Redundant latent dimensions (w//8, h//8)
    bucket_index: int              # Index in bucket list
    size_class: str               # Size classification (e.g., "small", "medium", "large")
    aspect_class: str            # Aspect ratio classification (e.g., "portrait", "landscape", "square")
    
    @classmethod
    def from_dims(cls, width: int, height: int, bucket_index: int) -> 'BucketInfo':
        """Create BucketInfo with full validation."""
        dimensions = BucketDimensions.from_pixels(width, height)
        
        # Validate dimension consistency
        if not dimensions.validate():
            raise ValueError(f"Invalid dimensions for bucket: {width}x{height}")
        
        # Classify bucket
        size_class = cls._classify_size(dimensions.total_pixels)
        aspect_class = cls._classify_aspect(dimensions.aspect_ratio)
        
        return cls(
            dimensions=dimensions,
            pixel_dims=(width, height),
            latent_dims=(width // 8, height // 8),
            bucket_index=bucket_index,
            size_class=size_class,
            aspect_class=aspect_class
        )
    
    @staticmethod
    def _classify_size(total_pixels: int) -> str:
        """Classify bucket by total pixels."""
        if total_pixels < 512 * 512:
            return "small"
        elif total_pixels < 1024 * 1024:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _classify_aspect(ratio: float) -> str:
        """Classify bucket by aspect ratio."""
        if np.isclose(ratio, 1.0, atol=0.1):
            return "square"
        elif ratio > 1.0:
            return "landscape"
        else:
            return "portrait"
    
    def validate(self) -> bool:
        """Comprehensive validation of all stored information."""
        try:
            # Validate dimensions object
            if not self.dimensions.validate():
                return False
            
            # Validate consistency between redundant storage
            checks = [
                self.pixel_dims == (self.dimensions.width, self.dimensions.height),
                self.latent_dims == (self.dimensions.width_latent, self.dimensions.height_latent),
                self.dimensions.aspect_ratio == self.pixel_dims[0] / self.pixel_dims[1],
                self._classify_size(self.dimensions.total_pixels) == self.size_class,
                self._classify_aspect(self.dimensions.aspect_ratio) == self.aspect_class
            ]
            return all(checks)
            
        except Exception:
            return False

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
    buckets: List[BucketInfo]
) -> BucketInfo:
    """Find closest bucket using comprehensive metrics."""
    w, h = original_size
    original_ratio = w / h
    original_pixels = w * h
    
    # Find closest bucket using multiple metrics
    closest_bucket = min(
        buckets,
        key=lambda b: (
            abs(b.total_pixels - original_pixels) / original_pixels * 0.5 +  # Area difference
            abs(b.aspect_ratio - original_ratio) * 2.0                       # Aspect ratio difference
        )
    )
    
    return closest_bucket

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
                bucket_indices[bucket_info["pixel_dims"]].append(idx)
            else:
                # Process new image
                img = Image.open(path)
                bucket_info = compute_bucket_dims(img.size, buckets)
                
                # Store comprehensive bucket information
                bucket_indices[bucket_info.pixel_dims].append(idx)
                
                # Update cache with full bucket information
                if cached_entry:
                    cached_entry["bucket_info"] = {
                        "pixel_dims": bucket_info.pixel_dims,
                        "latent_dims": bucket_info.latent_dims,
                        "aspect_ratio": bucket_info.aspect_ratio,
                        "total_pixels": bucket_info.total_pixels,
                        "total_latents": bucket_info.total_latents,
                        "bucket_index": bucket_info.bucket_index
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