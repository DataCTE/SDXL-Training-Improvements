"""High-performance preprocessing pipeline with extreme speedups."""
import time
import torch
import random
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
from src.core.memory.tensor import create_stream_context
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
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        use_pinned_memory: bool = True,
        enable_memory_tracking: bool = True,
        stream_timeout: float = 10.0
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            config: Configuration object
            latent_preprocessor: Optional latent preprocessor
            cache_manager: Optional cache manager
            is_train: Whether this is for training
            num_gpu_workers: Number of GPU workers for parallel processing
            num_cpu_workers: Number of CPU workers for preprocessing
            num_io_workers: Number of IO workers for data loading
            prefetch_factor: Prefetch factor for data loading
            use_pinned_memory: Whether to use pinned memory
            enable_memory_tracking: Whether to track memory usage
            stream_timeout: Timeout for CUDA stream operations
        """
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
        
        # Store worker configurations
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        
        # Create dedicated CUDA stream for this pipeline
        self.stream = torch.cuda.Stream(device=self.device_id) if torch.cuda.is_available() else None
        self.prefetch_factor = prefetch_factor
        self.use_pinned_memory = use_pinned_memory
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_timeout = stream_timeout
        self.stats = PipelineStats()
        # Initialize bucket-related attributes
        self.buckets = self.get_aspect_buckets(config)
        self.bucket_indices = []  # Will be populated when processing images

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """Get list of supported bucket dimensions from config.
        
        Args:
            config: Configuration object containing image settings
            
        Returns:
            List of (height, width) tuples for each bucket
        """
        return config.global_config.image.supported_dims

    def _assign_single_bucket(
        self,
        img: Union[str, Path, Image.Image],
        max_aspect_ratio: Optional[float] = None
    ) -> Tuple[int, Tuple[int, int]]:
        """Assign image to optimal bucket based on aspect ratio and size.
        
        Args:
            img: Image path or PIL Image object
            max_aspect_ratio: Optional override for max aspect ratio
                
        Returns:
            Tuple of (bucket_index, (height, width))
                
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            # Handle both PIL Image and path inputs
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
                
            w, h = img.size
            aspect_ratio = w / h
            img_area = w * h
                
            # Use config max_aspect_ratio if not overridden
            max_ar = max_aspect_ratio or self.config.global_config.image.max_aspect_ratio
                
            min_diff = float('inf')
            best_idx = 0
            best_bucket = self.buckets[0]
                
            for idx, bucket in enumerate(self.buckets):
                bucket_h, bucket_w = bucket
                bucket_ratio = bucket_w / bucket_h
                    
                # Skip buckets exceeding max aspect ratio
                if bucket_ratio > max_ar:
                    continue
                        
                # Calculate weighted difference score
                ratio_diff = abs(aspect_ratio - bucket_ratio)
                area_diff = abs(img_area - (bucket_w * bucket_h))
                    
                # Combined score favoring aspect ratio match
                total_diff = (ratio_diff * 2.0) + (area_diff / (1536 * 1536))
                    
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_idx = idx
                    best_bucket = bucket
                        
            return best_idx, best_bucket
                
        except Exception as e:
            logger.error(f"Error assigning bucket for {img}: {e}")
            # Return default bucket on error
            return 0, self.buckets[0]

    def get_bucket_info(self) -> Dict[str, Any]:
        """Get information about current bucket configuration.
        
        Returns:
            Dictionary containing bucket statistics and configuration
        """
        bucket_stats = {
            'total_buckets': len(self.buckets),
            'bucket_dimensions': self.buckets,
            'bucket_counts': {},
            'aspect_ratios': {},
            'total_images': len(self.bucket_indices) if hasattr(self, 'bucket_indices') else 0
        }
        
        # Calculate statistics if we have bucket assignments
        if hasattr(self, 'bucket_indices') and self.bucket_indices:
            for idx in self.bucket_indices:
                bucket_stats['bucket_counts'][idx] = bucket_stats['bucket_counts'].get(idx, 0) + 1
                h, w = self.buckets[idx]
                bucket_stats['aspect_ratios'][idx] = w / h
                
        return bucket_stats

    def validate_image_size(
        self,
        size: Tuple[int, int],
        min_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Validate image dimensions against configuration.
        
        Args:
            size: Tuple of (height, width)
            min_size: Optional minimum dimensions
            max_size: Optional maximum dimensions
            
        Returns:
            bool indicating if size is valid
        """
        min_size = min_size or self.config.global_config.image.min_size
        max_size = max_size or self.config.global_config.image.max_size
        
        h, w = size
        min_h, min_w = min_size
        max_h, max_w = max_size
        
        # Check absolute dimensions
        if h < min_h or w < min_w:
            return False
        if h > max_h or w > max_w:
            return False
            
        # Check aspect ratio
        aspect = w / h
        if aspect < 1/self.config.global_config.image.max_aspect_ratio:
            return False
        if aspect > self.config.global_config.image.max_aspect_ratio:
            return False
            
        return True
    
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



    def precompute_latents(
        self,
        image_paths: List[str],
        batch_size: int = 1,
        proportion_empty_prompts: float = 0.0,
        process_latents: bool = True,
        process_text_embeddings: bool = True,
        separate_passes: bool = True
    ) -> None:
        """Precompute and cache latents and text embeddings with cache verification."""
        if not self.latent_preprocessor or not self.cache_manager:
            logger.warning("Latent preprocessor or cache manager not available")
            return

        # Verify cache folders and get cached files
        image_latents_dir = Path(self.cache_manager.cache_dir) / "image"
        text_embeddings_dir = Path(self.cache_manager.cache_dir) / "text"
        
        # Check if directories exist
        image_latents_dir.mkdir(exist_ok=True)
        text_embeddings_dir.mkdir(exist_ok=True)
        
        # Get actual cached files
        cached_image_files = set(p.stem for p in image_latents_dir.glob("*.pt"))
        cached_text_files = set(p.stem for p in text_embeddings_dir.glob("*.pt"))
        
        # Load and verify current cache index
        try:
            old_index = self.cache_manager._load_cache_index()  # Load existing index for reference
            logger.info(f"Found existing cache index with {len(old_index.get('files', {}))} entries")
        except Exception as e:
            logger.warning(f"Could not load existing cache index: {e}")
            old_index = {"files": {}, "chunks": {}}
        
        # Build new index from actual files
        new_index = {"files": {}, "chunks": {}}
        
        for img_path in image_paths:
            base_name = Path(img_path).stem
            file_info = {}
            
            # Check for image latents
            if base_name in cached_image_files:
                file_info["latent_path"] = str(image_latents_dir / f"{base_name}.pt")
                file_info["type"] = "image"
                
            # Check for text embeddings
            if base_name in cached_text_files:
                file_info["text_path"] = str(text_embeddings_dir / f"{base_name}.pt")
                file_info["text_type"] = "text"
                
            if file_info:
                new_index["files"][str(img_path)] = file_info
        
        # Save rebuilt index
        self.cache_manager._save_cache_index(new_index)
        logger.info("Cache index rebuilt successfully")
        
        # First pass: Process image latents if requested
        if process_latents:
            missing_latents = []
            for img_path in image_paths:
                img_id = Path(img_path).stem
                if img_id not in cached_image_files:
                    missing_latents.append(img_path)

            if missing_latents:
                logger.info(f"Processing {len(missing_latents)} missing image latents out of {len(image_paths)} total")
                with tqdm(total=len(missing_latents), desc="Processing image latents") as pbar:
                    for i in range(0, len(missing_latents), batch_size):
                        batch_paths = missing_latents[i:i+batch_size]
                        
                        for img_path in batch_paths:
                            try:
                                processed = self._process_image(img_path)
                                if processed:
                                    metadata = {
                                        "path": img_path,
                                        "timestamp": time.time(),
                                        **processed.get("metadata", {})
                                    }
                                    
                                    self.cache_manager.save_preprocessed_data(
                                        image_latent=processed["image_latent"],
                                        text_embeddings=None,
                                        metadata=metadata,
                                        file_path=img_path
                                    )
                                    self.stats.successful += 1
                                    
                            except Exception as e:
                                self.stats.failed += 1
                                logger.error(f"Failed to process image {img_path}: {e}", exc_info=True)
                            finally:
                                pbar.update(1)
                        
                        if torch.cuda.is_available() and i % 100 == 0:
                            torch.cuda.empty_cache()
            else:
                logger.info("All image latents already cached")

        # Reset stats for second pass
        self.stats = PipelineStats()

        # Second pass: Process text embeddings
        if process_text_embeddings:
            missing_embeddings = []
            for img_path in image_paths:
                # Convert image path to text path properly by replacing the extension
                img_path = Path(img_path)
                txt_path = img_path.parent / f"{img_path.stem}.txt"
                
                if txt_path.exists():  # Only process if text file exists
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                            
                        if caption and not (proportion_empty_prompts > 0 and random.random() < proportion_empty_prompts):
                            img_id = img_path.stem
                            if img_id not in cached_text_files:
                                missing_embeddings.append({
                                    "caption": caption,
                                    "img_id": img_id,
                                    "img_path": str(img_path),
                                    "txt_path": str(txt_path)
                                })
                    except Exception as e:
                        logger.warning(f"Failed to read caption from {txt_path}: {e}")
                        continue

            if missing_embeddings:
                logger.info(f"Processing {len(missing_embeddings)} missing text embeddings")
                with tqdm(total=len(missing_embeddings), desc="Processing text embeddings") as pbar:
                    for i in range(0, len(missing_embeddings), batch_size):
                        batch_items = missing_embeddings[i:i+batch_size]
                        
                        for item in batch_items:
                            try:
                                embeddings = self.latent_preprocessor.encode_prompt([item["caption"]])
                                metadata = {
                                    "image_path": item["img_path"],
                                    "text_path": item["txt_path"],
                                    "timestamp": time.time(),
                                    "caption": item["caption"]
                                }

                                self.cache_manager.save_preprocessed_data(
                                    image_latent=None,
                                    text_embeddings=embeddings,
                                    metadata=metadata,
                                    file_path=item["img_id"],  # Use image ID for cache consistency
                                    caption=item["caption"]
                                )
                                self.stats.successful += 1

                            except Exception as e:
                                self.stats.failed += 1
                                logger.error(f"Failed to process caption from {item['txt_path']}: {e}", exc_info=True)
                            finally:
                                pbar.update(1)

                        if torch.cuda.is_available() and i % 100 == 0:
                            torch.cuda.empty_cache()
            else:
                logger.info("All text embeddings already cached")


    def get_valid_image_paths(self) -> List[str]:
        """Return list of valid image paths found during bucketing."""
        if not hasattr(self, 'valid_image_paths'):
            return []
        return self.valid_image_paths

    
    def resize_to_bucket(
        self,
        img: Image.Image,
        bucket_idx: Optional[int] = None
    ) -> Tuple[Image.Image, int]:
        """Resize image to fit target bucket dimensions.
        
        Args:
            img: PIL Image to resize
            bucket_idx: Optional specific bucket index to use
            
        Returns:
            Tuple of (resized image, used bucket index)
        """
        try:
            if bucket_idx is None:
                # Get bucket index and dimensions
                bucket_idx, _ = self._assign_single_bucket(img)
                
            # Get target dimensions from bucket
            target_h, target_w = self.buckets[bucket_idx]
            
            # Resize image
            resized_img = img.resize((target_w, target_h), Image.LANCZOS)
            
            return resized_img, bucket_idx
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            # Use first bucket as fallback
            target_h, target_w = self.buckets[0]
            return img.resize((target_w, target_h), Image.LANCZOS), 0

    def _process_image(self, img_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a single image."""
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Validate image size first
            if not self.validate_image_size(img.size):
                raise ValueError(f"Invalid image size: {img.size}")
                
            # Get bucket assignment and resize image
            resized_img, bucket_idx = self.resize_to_bucket(img)
            
            # Convert to tensor and process
            tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            
            # Process on GPU
            with torch.cuda.amp.autocast():
                latent = self.latent_preprocessor.encode_images(tensor.to(self.device))
            
            metadata = {
                "original_size": img.size,
                "bucket_size": self.buckets[bucket_idx],
                "bucket_index": bucket_idx,
                "path": str(img_path),
                "timestamp": time.time()
            }
            
            return {
                "image_latent": latent["image_latent"],
                "metadata": metadata
            }
            
        except Exception as e:
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
