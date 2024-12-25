"""High-performance preprocessing pipeline with extreme speedups."""
import time
import torch
import logging
from contextlib import contextmanager
from src.core.logging.logging import setup_logging

logger = setup_logging(__name__, level=logging.INFO)
import torch.backends.cudnn
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
from src.data.utils.paths import convert_windows_path, is_windows_path
from contextlib import nullcontext
from src.data.preprocessing.cache_manager import CacheManager
from src.data.preprocessing.latents import LatentPreprocessor
import numpy as np
from src.data.config import Config

class ProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass


# Use the logger already initialized above

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
        config: Config,
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        cache_manager: Optional[CacheManager] = None,
        is_train=True,
        num_gpu_workers=1,  # Keep this at 1 to prevent issues
        num_cpu_workers=4,
        num_io_workers=2,
        prefetch_factor=2,
        device_ids=None,  # This needs fixing
        use_pinned_memory=True,
        enable_memory_tracking=True,
        stream_timeout=10.0
    ):
        # Add performance tracking
        self.performance_stats = {
            'operation_times': {},
            'memory_usage': {},
            'errors': []
        }
        self.logger = logger  # Use the module-level logger
        self.action_history = {}
        # Basic initialization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            # Force single GPU usage
            self.device_id = 0  # Always use first GPU
            self.device = torch.device(f'cuda:{self.device_id}')
            # Set this device as default
            torch.cuda.set_device(self.device_id)
        else:
            self.device_id = None
            self.device = torch.device('cpu')
            
        self.config = config if config is not None else Config()
        self.latent_preprocessor = latent_preprocessor
        if self.latent_preprocessor:
            self.latent_preprocessor.device = self.device
        self.cache_manager = cache_manager
        self.is_train = is_train
        # Update num_gpu_workers to always be 1
        self.num_gpu_workers = 1
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        
        # Create dedicated CUDA stream for this pipeline
        self.stream = torch.cuda.Stream(device=self.device_id) if torch.cuda.is_available() else None
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids
        self.use_pinned_memory = use_pinned_memory
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_timeout = stream_timeout
        self.stats = PipelineStats()
        self.target_image_size = (1024, 1024)  # Define your target dimensions
        self.valid_image_paths = []

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """
        Generate aspect buckets based on the image configurations.

        Args:
            config: Configuration object containing image settings.

        Returns:
            List of tuples representing the bucket dimensions (height, width).
        """
        return config.global_config.image.supported_dims


    def _read_caption(self, img_path: Union[str, Path]) -> str:
        # Construct the path to the corresponding .txt file
        caption_path = Path(img_path).with_suffix('.txt')
        if not caption_path.exists():
            logger.warning(f"Caption file not found for image {img_path}. Using empty caption.")
            return ""
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return caption

    def group_images_by_aspect_ratio(self, image_paths: Union[str, Path, Config], tolerance: float = 0.05) -> Dict[str, List[str]]:
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
                        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff', '*.tif', '*.ppm', '*.pgm'):
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
                    # Use rounded aspect ratio as bucket key
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

        # Store valid image paths
        self.valid_image_paths = [path for paths in buckets.values() for path in paths]

        return buckets

    def generate_missing_captions_from_cache(self):
        """Generate missing caption files from cache index."""
        if not self.cache_manager:
            logger.warning("CacheManager is not available.")
            return

        cache_index = self.cache_manager.cache_index.get("files", {})
        for image_path_str in cache_index:
            image_path = Path(image_path_str)
            caption_path = image_path.with_suffix('.txt')

            # Skip if caption file already exists
            if caption_path.exists():
                continue

            # Load text embeddings from cache
            text_data = self.cache_manager.load_text_embeddings(image_path)
            if not text_data:
                logger.warning(f"No text embeddings found for {image_path}. Skipping.")
                continue

            # Extract caption from metadata
            caption = text_data.get("metadata", {}).get("caption", "")
            if not caption:
                logger.warning(f"No caption found in metadata for {image_path}. Skipping.")
                continue

            # Write the caption to a .txt file
            try:
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                logger.info(f"Created caption file: {caption_path}")
            except Exception as e:
                logger.error(f"Failed to write caption file {caption_path}: {e}")

    def get_processed_item(self, image_path: Union[str, Path], caption: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image and return the preprocessed data."""
        try:
            processed_data = {}

            if self.cache_manager:
                cached_data = self.cache_manager.load_preprocessed_data(image_path)
                if cached_data:
                    self.stats.cache_hits += 1
                    processed_data.update(cached_data)
                else:
                    self.stats.cache_misses += 1

            if "latent" not in processed_data:
                processed = self._process_image(image_path)
                if processed:
                    processed_data["latent"] = processed["latent"]
                    processed_data.setdefault("metadata", {}).update(processed.get("metadata", {}))
                else:
                    raise ProcessingError(f"Failed to process image: {image_path}")

            if caption is None:
                caption = self._read_caption(image_path)

            if "text_embeddings" not in processed_data:
                embeddings = self.latent_preprocessor.encode_prompt([caption])
                processed_data["text_embeddings"] = embeddings

            processed_data["text"] = caption

            if self.cache_manager:
                self.cache_manager.save_preprocessed_data(
                    latent_data=processed_data.get("latent"),
                    text_embeddings=processed_data.get("text_embeddings"),
                    metadata=processed_data.get("metadata", {}),
                    file_path=image_path,
                    caption=caption  # Pass caption
                )

            return processed_data

        except Exception as e:
            logger.error(f"Error processing item {image_path}: {e}")
            raise

    def assign_aspect_buckets(self, image_paths: Union[str, Path, Config], tolerance: float = 0.1) -> List[int]:
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
        batch_size=1,
        proportion_empty_prompts=0.0,
        process_latents=True,
        process_text_embeddings=True
    ):
        if not self.latent_preprocessor or not self.cache_manager or not self.is_train:
            return
        logger.info(f"Precomputing latents and embeddings for {len(image_paths)} items")
        to_process = []

        # Use the validated cache index
        cache_index = self.cache_manager.cache_index.get("files", {})
        cached_files = set(cache_index.keys())

        # Track memory usage
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial CUDA memory: {initial_memory/1024**2:.1f}MB")

        # First pass: identify what needs processing
        for path in tqdm(image_paths, desc="Checking cache status"):
            path_str = str(path)
            if path_str not in cached_files:
                to_process.append(path)
                continue
                
            file_info = cache_index.get(path_str)
            if not file_info:
                to_process.append(path)
                continue

            # Verify cached files exist
            latent_path = Path(file_info.get("latent_path", ""))
            text_path = Path(file_info.get("text_path", ""))
            
            if not latent_path.exists() or not text_path.exists():
                to_process.append(path)

        if not to_process:
            logger.info("All items already cached")
            return

        logger.info(f"Processing {len(to_process)} uncached items")

        # Process in batches with progress bar
        with tqdm(total=len(to_process), desc="Processing items") as pbar:
            for i in range(0, len(to_process), batch_size):
                batch_paths = to_process[i:i+batch_size]
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available() and i % 100 == 0:
                    torch.cuda.empty_cache()
                    current_memory = torch.cuda.memory_allocated()
                    logger.debug(f"CUDA memory: {current_memory/1024**2:.1f}MB")

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
                            self.stats.successful += 1
                    except Exception as e:
                        self.stats.failed += 1
                        logger.warning(f"Failed to process {img_path}: {e}")
                    finally:
                        pbar.update(1)

                # Force CUDA sync periodically
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.synchronize()

        # Final memory report
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            logger.info(f"Final CUDA memory: {final_memory/1024**2:.1f}MB")
            logger.info(f"Memory change: {(final_memory - initial_memory)/1024**2:.1f}MB")

    def _process_image(self, img_path):
        try:
            def gpu_process(tensor):
                tensor = tensor.to(self.device, non_blocking=True)
                vae_dtype = next(self.latent_preprocessor.model.vae.parameters()).dtype
                tensor = tensor.to(dtype=vae_dtype)
                return self.latent_preprocessor.encode_images(tensor)

            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_image_size, Image.LANCZOS)
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            
            # Process on GPU with proper stream management
            latent_output = self._process_on_gpu(gpu_process, tensor)
            
            metadata = {
                "original_size": img.size,
                "path": str(img_path),
                "timestamp": time.time(),
                "device_id": self.device_id if torch.cuda.is_available() else None
            }
            return {"latent": latent_output["latent"], "metadata": metadata}
        except Exception as e:
            self.stats.failed += 1
            logger.warning(f"Failed to process {img_path}: {e}")
            return None

    def encode_prompt(self, caption: str) -> Dict[str, torch.Tensor]:
        """Encode a text prompt into embeddings using the latent preprocessor.
        
        Args:
            caption: Text caption to encode
            
        Returns:
            Dictionary containing text embeddings
        """
        try:
            if not self.latent_preprocessor:
                raise ValueError("Latent preprocessor not initialized")
            
            embeddings = self.latent_preprocessor.encode_prompt([caption])
            return {
                "text_embeddings": embeddings,
                "metadata": {"caption": caption, "timestamp": time.time()}
            }
        except Exception as e:
            logger.error(f"Failed to encode prompt: {e}")
            raise
    @contextmanager
    def track_memory_usage(self, operation: str):
        """Context manager for tracking memory usage during operations."""
        try:
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            yield
            
        finally:
            duration = time.time() - start_time
            memory_stats = {}
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_stats.update({
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'peak_memory': peak_memory,
                    'memory_change': end_memory - start_memory
                })
                
            self._log_action(operation, {
                'duration': duration,
                'memory_stats': memory_stats
            })

    def _process_on_gpu(self, func, *args, **kwargs):
        """Execute function on GPU with proper stream management."""
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
            
        try:
            with torch.cuda.device(self.device_id):
                with create_stream_context(self.stream):
                    result = func(*args, **kwargs)
                    if self.stream:
                        self.stream.synchronize()
                    return result
        except Exception as e:
            logger.error(f"GPU processing error: {str(e)}")
            raise ProcessingError("GPU processing failed", {
                'device_id': self.device_id,
                'cuda_memory': torch.cuda.memory_allocated(self.device_id),
                'error': str(e)
            })

    def _log_action(self, operation: str, stats: Dict[str, Any]):
        """Log operation statistics."""
        if operation not in self.performance_stats['operation_times']:
            self.performance_stats['operation_times'][operation] = []
        self.performance_stats['operation_times'][operation].append(stats)
