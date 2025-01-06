"""Data handling components for SDXL training."""
from .config import Config
from .utils.paths import convert_windows_path, is_windows_path, is_wsl
from .dataset import AspectBucketDataset, create_dataset
from .preprocessing.cache_manager import CacheManager
from .preprocessing.tag_weighter import (
    TagWeighter, 
    create_tag_weighter,
    create_tag_weighter_with_index,
    preprocess_dataset_tags
)
from .preprocessing.bucket_utils import (
    generate_buckets,
    compute_bucket_dims,
    group_images_by_bucket,
    log_bucket_statistics
)
from .preprocessing.bucket_types import BucketInfo, BucketDimensions
from .preprocessing.exceptions import (
    PreprocessingError,
    DataLoadError,
    PipelineConfigError,
    GPUProcessingError,
    CacheError,
    DtypeError,
    DALIError,
    TensorValidationError,
    StreamError,
    MemoryError,
    TagProcessingError
)

__all__ = [
    # Core components
    "Config",
    "AspectBucketDataset",
    "create_dataset",
    
    # Cache management
    "CacheManager",
    
    # Tag processing
    "TagWeighter",
    "create_tag_weighter",
    "create_tag_weighter_with_index",
    "preprocess_dataset_tags",
    
    # Path utilities
    "convert_windows_path",
    "is_windows_path",
    "is_wsl",
    
    # Bucket handling
    "generate_buckets",
    "compute_bucket_dims",
    "group_images_by_bucket",
    "log_bucket_statistics",
    "BucketInfo",
    "BucketDimensions",
    
    # Exceptions
    "PreprocessingError",
    "DataLoadError", 
    "PipelineConfigError",
    "GPUProcessingError",
    "CacheError",
    "DtypeError",
    "DALIError",
    "TensorValidationError",
    "StreamError",
    "MemoryError",
    "TagProcessingError"
]
