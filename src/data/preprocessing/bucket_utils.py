"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple, Dict, TYPE_CHECKING
import logging
from collections import defaultdict
from tqdm import tqdm
from src.data.config import Config
import torch

if TYPE_CHECKING:
    from src.data.preprocessing.cache_manager import CacheManager

logger = logging.getLogger(__name__)

def get_bucket_dims_from_latents(latent_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Convert VAE latent dimensions to bucket dimensions (source of truth)."""
    # VAE latents have shape (4, H/8, W/8)
    # So we multiply spatial dims by 8 to get original dimensions
    _, h, w = latent_shape
    return (w * 8, h * 8)

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
    """Compute bucket dimensions that will result in correct VAE latent sizes."""
    w, h = original_size
    
    # Scale dimensions to be VAE compatible (multiple of 8)
    w = (w + 7) // 8 * 8
    h = (h + 7) // 8 * 8
    
    # Find closest bucket by aspect ratio
    aspect_ratio = w / h
    _, bucket_dims = min(
        [(abs(aspect_ratio - (bw/bh)), (bw,bh)) for bw,bh in buckets]
    )
    
    return bucket_dims

def validate_and_fix_bucket_dims(
    computed_bucket: Tuple[int, int],
    latents: "torch.Tensor",
    image_path: str
) -> Tuple[int, int]:
    """Validate bucket dimensions against VAE latents and use VAE dims if mismatch."""
    actual_dims = get_bucket_dims_from_latents(latents.shape)
    
    if computed_bucket != actual_dims:
        logger.debug(
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
    cache_manager: "CacheManager"
) -> Dict[Tuple[int, int], List[int]]:
    """Group image indices by their VAE latent dimensions to ensure compatible batches."""
    bucket_indices = defaultdict(list)
    
    logger.info("Grouping images by VAE latent dimensions...")
    for idx, image_path in enumerate(tqdm(image_paths, desc="Grouping images")):
        cache_key = cache_manager.get_cache_key(image_path)
        cache_entry = cache_manager.cache_index["entries"].get(cache_key)
        
        if not cache_entry:
            logger.warning(f"Missing cache entry for {image_path}, skipping")
            continue
            
        try:
            cached_data = cache_manager.load_tensors(cache_key)
            if cached_data is None or "vae_latents" not in cached_data:
                logger.warning(f"Missing VAE latents for {image_path}, skipping")
                continue
            
            # Group by actual latent dimensions (H/8, W/8)
            latent_bucket = get_latent_bucket_key(cached_data["vae_latents"])
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