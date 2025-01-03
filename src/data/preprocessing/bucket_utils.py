"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import ImageConfig
import torch

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def get_bucket_dims_from_latents(latent_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Get latent dimensions directly (H/8, W/8)."""
    _, h, w = latent_shape
    return (w, h)  # Return latent dimensions directly, don't multiply by 8

def generate_buckets(config: "ImageConfig") -> List[Tuple[int, int]]:
    """Generate bucket dimensions from config."""
    buckets = []
    
    # Add supported dimensions from config
    for dims in config.supported_dims:
        # Convert to latent dimensions
        buckets.append((dims[0] // 8, dims[1] // 8))
    
    # Add intermediate buckets based on bucket_step
    step = config.bucket_step // 8  # Convert step to latent space
    min_w, min_h = config.min_size[0] // 8, config.min_size[1] // 8
    max_w, max_h = config.max_size[0] // 8, config.max_size[1] // 8
    
    for w in range(min_w, max_w + 1, step):
        for h in range(min_h, max_h + 1, step):
            if validate_aspect_ratio(w, h, config.max_aspect_ratio):
                bucket = (w, h)
                if bucket not in buckets:
                    buckets.append(bucket)
    
    return sorted(buckets, key=lambda x: (x[0] * x[1], x[0]))  # Sort by area, then width

def compute_bucket_dims(original_size: Tuple[int, int], buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Convert original size to latent dimensions and find closest bucket."""
    # Convert to latent dimensions
    w = original_size[0] // 8  # Integer division to get latent size
    h = original_size[1] // 8
    
    # Find closest bucket by total area and aspect ratio
    target_area = w * h
    target_ratio = w / h
    
    def bucket_distance(bucket: Tuple[int, int]) -> float:
        bw, bh = bucket
        area_diff = abs(bw * bh - target_area) / target_area
        ratio_diff = abs((bw/bh) - target_ratio)
        return area_diff + ratio_diff
    
    bucket_dims = min(buckets, key=bucket_distance)
    return bucket_dims

def validate_and_fix_bucket_dims(
    computed_bucket: Tuple[int, int],
    latents: "torch.Tensor",
    image_path: str
) -> Tuple[int, int]:
    """Validate bucket dimensions against VAE latents."""
    actual_dims = get_bucket_dims_from_latents(latents.shape)
    
    if computed_bucket != actual_dims:
        logger.warning(
            f"Bucket dimension mismatch for {image_path}: "
            f"computed {computed_bucket}, VAE latents indicate {actual_dims}. "
            f"Using VAE dimensions."
        )
        return actual_dims
    return computed_bucket

def validate_aspect_ratio(width: int, height: int, max_ratio: float) -> bool:
    """Validate if the aspect ratio is within acceptable bounds."""
    aspect_ratio = width / height
    return 1/max_ratio <= aspect_ratio <= max_ratio

def get_latent_bucket_key(latents: "torch.Tensor") -> Tuple[int, int]:
    """Get bucket key from latent dimensions."""
    _, h, w = latents.shape
    # Return latent dimensions directly (don't multiply by 8)
    return (w, h)  # These are already in latent space (H/8, W/8)

def group_images_by_bucket(
    image_paths: List[str], 
    cache_manager: "CacheManager",
    auto_rebuild: bool = True
) -> Dict[Tuple[int, int], List[int]]:
    """Group images by their latent dimensions, handling first runs and partial caches."""
    bucket_indices = defaultdict(list)
    uncached_paths = cache_manager.get_uncached_paths(image_paths)
    
    # Handle uncached images
    if uncached_paths:
        if auto_rebuild:
            logger.info(f"Found {len(uncached_paths)} uncached images - rebuilding cache...")
            cache_manager.rebuild_cache_index()
        else:
            logger.warning(
                f"Found {len(uncached_paths)} uncached images. "
                "Run cache verification/rebuild before training."
            )
    
    # Generate buckets from config
    image_config = cache_manager.config.global_config.image
    buckets = generate_buckets(image_config)
    default_bucket = (image_config.target_size[0] // 8, image_config.target_size[1] // 8)
    
    # Process each image
    for idx, image_path in enumerate(tqdm(image_paths, desc="Grouping images")):
        cache_key = cache_manager.get_cache_key(image_path)
        cache_entry = cache_manager.cache_index["entries"].get(cache_key)
        
        if image_path in uncached_paths or not cache_entry:
            bucket_indices[default_bucket].append(idx)
            continue
            
        try:
            cached_data = cache_manager.load_tensors(cache_key)
            if cached_data and "vae_latents" in cached_data:
                latents = cached_data["vae_latents"]
                actual_dims = get_latent_bucket_key(latents)
                bucket_dims = compute_bucket_dims(
                    (actual_dims[0] * 8, actual_dims[1] * 8),
                    buckets
                )
                bucket_indices[bucket_dims].append(idx)
            else:
                bucket_indices[default_bucket].append(idx)
        except Exception as e:
            logger.warning(f"Failed to process {image_path}: {e}")
            bucket_indices[default_bucket].append(idx)
    
    log_bucket_statistics(bucket_indices, len(image_paths))
    return dict(bucket_indices)

def log_bucket_statistics(bucket_indices: Dict[Tuple[int, int], List[int]], total_images: int):
    """Log statistics about bucket distribution."""
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