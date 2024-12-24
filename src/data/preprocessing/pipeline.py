"""High-performance preprocessing pipeline with extreme speedups."""
import logging
import time
import torch
import torch.backends.cudnn
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from src.data.utils.paths import convert_windows_path, is_windows_path
from contextlib import nullcontext
from src.data.preprocessing.cache_manager import CacheManager
from src.data.preprocessing.latents import LatentPreprocessor

class ProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass
import numpy as np
from src.data.config import Config


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
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
        self.config = config if config is not None else Config()
        self.latent_preprocessor = latent_preprocessor
        self.cache_manager = cache_manager
        self.is_train = is_train
        self.num_gpu_workers = 0  # Disable GPU multi-workers
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
        self.target_image_size = (512, 512)  # Define your target dimensions
        self._init_pools()
        # Disable torch.compile for now due to logging issues

    def _init_pools(self):
        self.gpu_pool = None  # Disable GPU worker pool
        self.cpu_pool = ProcessPoolExecutor(
            max_workers=self.num_cpu_workers,
            mp_context=mp.get_context('spawn')  # Specify 'spawn' start method
        )
        self.io_pool = ThreadPoolExecutor(max_workers=self.num_io_workers)

    def _read_caption(self, img_path: Union[str, Path]) -> str:
        base_name = Path(img_path).stem
        # Construct the path to the corresponding .txt file
        caption_path = Path(img_path).with_suffix('.txt')
        if not caption_path.exists():
            logger.warning(f"Caption file not found for image {img_path}. Using empty caption.")
            return ""
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return caption
        """Group images into buckets based on aspect ratio for efficient batch processing.

        Args:
            image_paths: List of paths to images, single path string/Path, or Config object containing paths
            tolerance: Tolerance for aspect ratio differences (default: 0.05)

        Returns:
            Dict mapping aspect ratio strings to lists of image paths
        """
        # Handle various input types
        if isinstance(image_paths, Config):
            # Extract paths from Config
            paths = []
            if hasattr(image_paths.data, 'train_data_dir'):
                train_dirs = image_paths.data.train_data_dir
                if isinstance(train_dirs, (str, Path)):
                    train_dirs = [train_dirs]
                
                # Scan directories for image files
                for dir_path in train_dirs:
                    dir_path = Path(convert_windows_path(dir_path) if is_windows_path(dir_path) else dir_path)
                    if dir_path.exists() and dir_path.is_dir():
                        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                            paths.extend(str(convert_windows_path(p)) for p in dir_path.glob(ext))
                    else:
                        logger.warning(f"Training directory does not exist or is not a directory: {dir_path}")
                
                if not paths:
                    logger.warning(f"No image files found in training directories: {train_dirs}")
            image_paths = paths
        elif isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        elif not isinstance(image_paths, (list, tuple)):
            raise ValueError(f"image_paths must be a string, Path, list, tuple or Config object, got {type(image_paths)}")

        buckets = {}
        for path in image_paths:
            if not isinstance(path, (str, Path)):
                logger.warning(f"Skipping invalid path type: {type(path)}")
                continue

            try:
                path_str = str(convert_windows_path(path) if is_windows_path(path) else path)
                if not Path(path_str).exists():
                    logger.warning(f"Image path does not exist: {path_str}")
                    continue

                with Image.open(path_str) as img:
                    w, h = img.size
                    aspect = w / h
                    # Use actual dimensions as bucket key
                    bucket_key = f"{w}_{h}"
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

        # Store valid image paths
        self.valid_image_paths = [path for paths in buckets.values() for path in paths]

        return buckets

    def get_processed_item(self, image_path: Union[str, Path], caption: Optional[str] = None, cache_manager: Optional['CacheManager'] = None, latent_preprocessor: Optional['LatentPreprocessor'] = None) -> Dict[str, Any]:
        """Process a single dataset item.
        
        Args:
            image_path: Path to image file
            caption: Image caption
            cache_manager: Optional cache manager
            latent_preprocessor: Optional latent preprocessor
            
        Returns:
            Dictionary containing processed data
        """
        try:
            # Check cache first if available
            if cache_manager:
                cached_data = cache_manager.load_preprocessed_data(image_path)
                if cached_data:
                    self.stats.cache_hits += 1
                    return cached_data
                    
            # Process image if not cached
            processed = self._process_image(image_path)
            if processed is None:
                raise ProcessingError(f"Failed to process image: {image_path}")
                
            # Read caption if not provided
            if caption is None:
                caption = self._read_caption(image_path)

            # Process text embeddings if needed
            text_embeddings = None
            if latent_preprocessor:
                text_embeddings = latent_preprocessor.encode_prompt([caption])

            # Create processed data dictionary
            processed_data = {
                "latent": processed["latent"],
                "text_embeddings": text_embeddings,
                "metadata": processed.get("metadata", {}),
                "text": caption
            }
            if cache_manager and latent_preprocessor:
                cache_manager.save_preprocessed_data(
                    latent_data=processed["latent"],
                    text_embeddings=processed.get("text_embeddings"),
                    metadata=processed.get("metadata", {}),
                    file_path=image_path
                )
                self.stats.cache_misses += 1
                
            return processed
            
        except Exception as e:
            logger.error(f"Error processing item {image_path}: {e}")
            raise

    def assign_aspect_buckets(self, image_paths: List[Union[str, Path]], tolerance: float = 0.1) -> List[int]:
        """Assign images to aspect ratio buckets.
        
        Args:
            image_paths: List of paths to images
            tolerance: Tolerance for aspect ratio differences (default: 0.1)
            
        Returns:
            List of bucket indices for each image
        """
        buckets = {}
        bucket_indices = []
        
        for path in image_paths:
            try:
                path_str = str(convert_windows_path(path) if is_windows_path(path) else path)
                if not Path(path_str).exists():
                    logger.warning(f"Image path does not exist: {path_str}")
                    bucket_indices.append(0)  # Default to first bucket
                    continue
                    
                with Image.open(path_str) as img:
                    w, h = img.size
                    aspect = w / h
                    # Round aspect ratio to nearest tolerance interval
                    bucket_key = f"{round(aspect / tolerance) * tolerance:.2f}"
                    if bucket_key not in buckets:
                        buckets[bucket_key] = len(buckets)
                    bucket_indices.append(buckets[bucket_key])
                    
            except Exception as e:
                logger.warning(f"Failed to process {path} for bucketing: {e}")
                bucket_indices.append(0)  # Default to first bucket
                continue
                
        return bucket_indices
        
    def get_valid_image_paths(self) -> List[str]:
        """Return list of valid image paths found during bucketing."""
        if not hasattr(self, 'valid_image_paths'):
            return []
        return self.valid_image_paths

    def precompute_latents(
        self,
        image_paths,
        captions,
        latent_preprocessor,
        batch_size=1,
        proportion_empty_prompts=0.0,
        process_latents=True,
        process_text_embeddings=True
    ):
        if not latent_preprocessor or not self.cache_manager or not self.is_train:
            return
        logger.info(f"Precomputing latents and embeddings for {len(image_paths)} items")
        to_process = []

        # Get cache index if available
        cache_index = self.cache_manager.cache_index.get("files", {})
        cached = set(cache_index.keys())

        for path in image_paths:
            if path not in cached:
                to_process.append(path)

        for i in range(0, len(to_process), batch_size):
            batch_paths = to_process[i:i+batch_size]
            batch_caps = [captions[image_paths.index(p)] for p in batch_paths]
            for img_path in batch_paths:
                try:
                    latent_data = None
                    text_embeddings = None
                    metadata = {"path": img_path, "timestamp": time.time()}

                    if process_latents:
                        processed = self._process_image(img_path)
                        if processed:
                            latent_data = processed["latent"]
                            metadata.update(processed.get("metadata", {}))

                    if process_text_embeddings:
                        caption = self._read_caption(img_path)
                        embeddings = self.latent_preprocessor.encode_prompt([caption])
                        text_embeddings = embeddings

                    # Save to cache
                    if latent_data or text_embeddings:
                        self.cache_manager.save_preprocessed_data(
                            latent_data=latent_data,
                            text_embeddings=text_embeddings,
                            metadata=metadata,
                            file_path=img_path
                        )
                        self.stats.cache_misses += 1
                        self.stats.successful += 1
                except Exception as e:
                    self.stats.failed += 1
                    logger.warning(f"Failed to precompute data for {img_path}: {e}")

    def _process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            # Do not resize the image
            metadata = {"original_size": img.size, "path": str(img_path), "timestamp": time.time()}
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            if torch.cuda.is_available():
                tensor = tensor.cuda(non_blocking=True)
            if self.latent_preprocessor:
                # Get the VAE data type from the latent_preprocessor
                vae_dtype = next(self.latent_preprocessor.model.vae.parameters()).dtype
                tensor = tensor.to(dtype=vae_dtype)
                latent = self.latent_preprocessor.encode_images(tensor)
            else:
                latent = tensor
            return {"latent": latent, "metadata": metadata}
        except Exception as e:
            self.stats.failed += 1
            logger.warning(f"Failed to process {img_path}: {e}")
            return None
