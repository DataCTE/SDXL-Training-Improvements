"""High-performance preprocessing pipeline with extreme speedups."""
import logging
import time
import torch
import torch.backends.cudnn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from dataclasses import dataclass
from src.data.utils.paths import convert_windows_path, is_windows_path
from contextlib import nullcontext
import numpy as np
from src.data.config import Config

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)

@dataclass
class PipelineStats:
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_oom_events: int = 0
    stream_sync_failures: int = 0
    dtype_conversion_errors: int = 0

class PreprocessingPipeline:
    def __init__(
        self,
        config,
        latent_preprocessor=None,
        cache_manager=None,
        is_train=True,
        num_gpu_workers=1,
        num_cpu_workers=4,
        num_io_workers=2,
        prefetch_factor=2,
        device_ids=None,
        use_pinned_memory=True,
        enable_memory_tracking=True,
        stream_timeout=10.0
    ):
        # Basic initialization
        self.config = config if config is not None else Config()
        self.latent_preprocessor = latent_preprocessor
        self.cache_manager = cache_manager
        self.is_train = is_train
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.use_pinned_memory = use_pinned_memory
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_timeout = stream_timeout
        self.stats = PipelineStats()
        self.input_queue = Queue(maxsize=prefetch_factor * num_gpu_workers)
        self.output_queue = Queue(maxsize=prefetch_factor * num_gpu_workers)
        self._init_pools()
        # Disable torch.compile for now due to logging issues

    def _init_pools(self):
        self.gpu_pool = (ThreadPoolExecutor(max_workers=self.num_gpu_workers) if torch.cuda.is_available() else None)
        self.cpu_pool = ProcessPoolExecutor(max_workers=self.num_cpu_workers)
        self.io_pool = ThreadPoolExecutor(max_workers=self.num_io_workers)

    def get_aspect_buckets(self, image_paths: Union[List[Union[str, Path]], str, Path, Config], tolerance: float = 0.1) -> Dict[str, List[str]]:
        """Group images into buckets based on aspect ratio for efficient batch processing.
        
        Args:
            image_paths: List of paths to images, single path string/Path, or Config object containing paths
            tolerance: Tolerance for aspect ratio differences (default: 0.1)
            
        Returns:
            Dict mapping aspect ratio strings to lists of image paths
        """
        # Handle Config object input
        if isinstance(image_paths, Config):
            paths = []
            if hasattr(image_paths.data, 'train_data_dir'):
                train_dirs = image_paths.data.train_data_dir
                if isinstance(train_dirs, (str, Path)):
                    train_dirs = [train_dirs]
                
                # Scan directories for image files
                for dir_path in train_dirs:
                    # Convert Windows paths if needed
                    dir_path = Path(convert_windows_path(dir_path) if is_windows_path(dir_path) else dir_path)
                    if dir_path.exists() and dir_path.is_dir():
                        # Find all image files in directory
                        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                            paths.extend(str(convert_windows_path(p)) for p in dir_path.glob(ext))
                    else:
                        logger.warning(f"Training directory does not exist or is not a directory: {dir_path}")
                
                if not paths:
                    logger.warning(f"No image files found in training directories: {train_dirs}")
            image_paths = paths
        # Convert single path to list
        elif isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        elif not isinstance(image_paths, (list, tuple)):
            raise ValueError(f"image_paths must be a string, Path, list, tuple or Config object, got {type(image_paths)}")
            
        buckets = {}
        for path in image_paths:
            # Skip if path is not a string or Path object
            if not isinstance(path, (str, Path)):
                logger.warning(f"Skipping invalid path type: {type(path)}")
                continue
                
            try:
                # Convert Windows paths if needed
                path_str = str(convert_windows_path(path) if is_windows_path(path) else path)
                if not Path(path_str).exists():
                    logger.warning(f"Image path does not exist: {path_str}")
                    continue
                    
                with Image.open(path_str) as img:
                    w, h = img.size
                    aspect = w / h
                    # Round aspect ratio to nearest tolerance interval
                    bucket_key = f"{round(aspect / tolerance) * tolerance:.2f}"
                    if bucket_key not in buckets:
                        buckets[bucket_key] = []
                    buckets[bucket_key].append(path_str)
                    self.stats.successful += 1
            except Exception as e:
                logger.warning(f"Failed to process {path} for bucketing: {e}")
                self.stats.failed += 1
                continue
        
        if not buckets:
            logger.warning("No valid images found for bucketing")
            return {}
            
        # Store valid image paths for dataset access
        self.valid_image_paths = []
        for paths in buckets.values():
            self.valid_image_paths.extend(paths)
            
        return buckets

    def assign_aspect_buckets(self, image_paths: List[Union[str, Path]], tolerance: float = 0.1) -> Dict[str, List[str]]:
        """Alias for get_aspect_buckets to maintain compatibility."""
        return self.get_aspect_buckets(image_paths, tolerance)
        
    def get_valid_image_paths(self) -> List[str]:
        """Return list of valid image paths found during bucketing."""
        if not hasattr(self, 'valid_image_paths'):
            return []
        return self.valid_image_paths

    def precompute_latents(self, image_paths, captions, latent_preprocessor, batch_size=1, proportion_empty_prompts=0.0):
        if not latent_preprocessor or not self.cache_manager or not self.is_train:
            return
        logger.info(f"Precomputing {len(image_paths)} latents")
        to_process = []
        
        # Get cache index if available
        cache_index = {}
        if self.cache_manager and hasattr(self.cache_manager, 'cache_index'):
            cache_index = self.cache_manager.cache_index.get("files", {})
        cached = set(cache_index.keys())
        for path in image_paths:
            if path not in cached: to_process.append(path)
            else: self.stats.cache_hits += 1
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i+batch_size]
            batch_caps = [captions[image_paths.index(p)] for p in batch]
            for img_path, cap in zip(batch, batch_caps):
                try:
                    processed = self._process_image(img_path)
                    if processed:
                        self.cache_manager.save_preprocessed_data(
                            latent_data=processed["latent"],
                            text_embeddings=processed.get("text_embeddings"),
                            metadata=processed.get("metadata", {}),
                            file_path=img_path
                        )
                        self.stats.cache_misses += 1
                        self.stats.successful += 1
                except Exception as e:
                    self.stats.failed += 1
                    logger.warning(e)

    def _process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            metadata = {"original_size": img.size, "path": str(img_path), "timestamp": time.time()}
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            if torch.cuda.is_available():
                tensor = tensor.cuda(non_blocking=True).to(dtype=torch.float16)
            if self.latent_preprocessor:
                latent = self.latent_preprocessor.encode_images(tensor)
            else:
                latent = tensor
            return {"latent": latent, "metadata": metadata}
        except Exception as e:
            self.stats.failed += 1
            logger.warning(f"Failed to process {img_path}: {e}")
            return None
