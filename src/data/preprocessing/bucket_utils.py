"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config
import torch

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def get_bucket_dims_from_latents(latent_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Get latent dimensions directly (H/8, W/8)."""
    _, h, w = latent_shape
    return (w, h)  # Return latent dimensions directly, don't multiply by 8

def generate_buckets(config: Optional[Config]) -> List[Tuple[int, int]]:
    """Generate bucket sizes in VAE latent space (H/8, W/8)."""
    if not config:
        # Return default buckets if no config provided
        default_buckets = [
            (64, 64),    # 512x512 
            (64, 96),    # 512x768
            (96, 64),    # 768x512
            (128, 128),  # 1024x1024
            (128, 192),  # 1024x1536
            (192, 128),  # 1536x1024
        ]
        logger.warning("No config provided, using default buckets")
        return default_buckets
        
    buckets = []
    
    # Convert config dimensions to latent space (divide by 8)
    min_size = (config.global_config.image.min_size[0] // 8,
                config.global_config.image.min_size[1] // 8)
    max_size = (config.global_config.image.max_size[0] // 8,
                config.global_config.image.max_size[1] // 8)
    step = max(config.global_config.image.bucket_step // 8, 1)
    
    # Convert supported dims to latent space
    supported_latent_dims = [
        (w // 8, h // 8) 
        for w, h in config.global_config.image.supported_dims
    ]
    buckets.extend(supported_latent_dims)
    
    # Generate additional buckets in latent space
    for h in range(min_size[1], max_size[1] + step, step):
        for w in range(min_size[0], max_size[0] + step, step):
            if max(w/h, h/w) <= config.global_config.image.max_aspect_ratio:
                bucket = (w, h)  # Already in latent space
                if bucket not in buckets:
                    buckets.append(bucket)
    
    return sorted(buckets)

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
    captions: List[str],
    cache_manager: "CacheManager",
    auto_rebuild: bool = True
) -> Dict[Tuple[int, int], List[int]]:
    """Group image indices by their VAE latent dimensions to ensure compatible batches."""
    bucket_indices = defaultdict(list)
    
    # First pass - check cache status
    missing_count = 0
    for image_path in image_paths:
        cache_key = cache_manager.get_cache_key(image_path)
        if cache_key not in cache_manager.cache_index["entries"]:
            missing_count += 1
    
    # If there are missing entries and auto_rebuild is enabled
    if missing_count > 0:
        if auto_rebuild:
            logger.warning(
                f"Found {missing_count} uncached images. "
                "Rebuilding cache before proceeding..."
            )
            cache_manager.verify_and_rebuild_cache(image_paths, captions)
        else:
            logger.warning(
                f"Found {missing_count} uncached images. "
                "Run cache verification/rebuild before training."
            )
    
    # Group images by bucket
    logger.info("Grouping images by VAE latent dimensions...")
    for idx, image_path in enumerate(tqdm(image_paths, desc="Grouping images")):
        cache_key = cache_manager.get_cache_key(image_path)
        cache_entry = cache_manager.cache_index["entries"].get(cache_key)
        
        if not cache_entry:
            continue
            
        try:
            cached_data = cache_manager.load_tensors(cache_key)
            if cached_data is None or "vae_latents" not in cached_data:
                continue
            
            # Validate latent dimensions
            latents = cached_data["vae_latents"]
            if len(latents.shape) != 3 or latents.shape[0] != 4:
                continue
                
            # Group by actual latent dimensions
            latent_bucket = get_latent_bucket_key(latents)
            bucket_indices[latent_bucket].append(idx)
            
            # Store latent dimensions in cache
            if cache_entry.get("latent_dims") != latent_bucket:
                cache_entry["latent_dims"] = latent_bucket
                cache_entry["needs_save"] = True
                cache_manager.cache_index["entries"][cache_key] = cache_entry
                cache_manager.cache_index["needs_save"] = True
                
        except Exception as e:
            logger.warning(f"Failed to process {image_path}: {e}")
            continue
    
    if not bucket_indices:
        raise RuntimeError("No valid images found in cache. Please preprocess the dataset first.")
    
    # Save any updates to cache index
    if cache_manager.cache_index.get("needs_save", False):
        cache_manager._save_index()
    
    # Log bucket distribution
    total_images = sum(len(indices) for indices in bucket_indices.values())
    logger.info(f"\nFound {len(bucket_indices)} distinct latent dimension groups:")
    for (w, h), indices in sorted(bucket_indices.items(), key=lambda x: len(x[1]), reverse=True):
        logger.info(f"Latent dims {w}x{h}: {len(indices)} images ({len(indices)/total_images*100:.1f}%)")
    
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