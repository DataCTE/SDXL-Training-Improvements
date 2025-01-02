"""Bucket calculation utilities for SDXL training."""
from typing import List, Tuple
from src.data.config import Config

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
    """Compute bucket dimensions for an image size."""
    w, h = original_size
    aspect_ratio = w / h
    bucket_ratios = [(bw/bh, (bw,bh)) for bw,bh in buckets]
    _, bucket_dims = min(
        [(abs(aspect_ratio - ratio), dims) for ratio, dims in bucket_ratios]
    )
    return bucket_dims 