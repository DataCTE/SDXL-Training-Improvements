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
            
            for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
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
                    best_bucket = (bucket_h, bucket_w)
                    
            return best_idx, best_bucket
            
        except Exception as e:
            logger.error(f"Error assigning bucket for {img}: {str(e)}")
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
        if bucket_idx is None:
            bucket_idx = self._assign_single_bucket(img, self.buckets, self.config.global_config.image.max_aspect_ratio)
            
        target_size = self.buckets[bucket_idx]
        resized_img = img.resize(target_size, Image.LANCZOS)
        
        return resized_img, bucket_idx


    def precompute_latents(
        self,
        image_paths: List[str],
        batch_size: int = 1,
        proportion_empty_prompts: float = 0.0,
        process_latents: bool = True,
        process_text_embeddings: bool = True
    ) -> None:
        """Precompute and cache latents and text embeddings with comprehensive processing.

        Args:
            image_paths: List of image paths to process
            batch_size: Batch size for processing
            proportion_empty_prompts: Proportion of prompts to leave empty
            process_latents: Whether to process image latents
            process_text_embeddings: Whether to process text embeddings
        """
        if not self.latent_preprocessor or not self.cache_manager:
            logger.warning("Latent preprocessor or cache manager not available")
            return

        logger.info(f"Starting comprehensive preprocessing for {len(image_paths)} items")

        # Track memory usage
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial CUDA memory: {initial_memory/1024**2:.1f}MB")

        # Validate cache and identify items needing processing
        cache_index = self.cache_manager.cache_index.get("files", {})
        to_process = []
        missing_captions = []

        for path in tqdm(image_paths, desc="Analyzing items"):
            path_str = str(path)
            caption_path = Path(path).with_suffix('.txt')
            cached_data = self.cache_manager.get_cached_item(path)

            needs_processing = False

            # Check if caption file needs to be generated
            if not caption_path.exists() and cached_data and "metadata" in cached_data:
                caption = cached_data["metadata"].get("caption", "")
                if caption:
                    try:
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                        logger.debug(f"Generated caption file for {path}")
                    except Exception as e:
                        logger.warning(f"Failed to write caption file {caption_path}: {e}")
                        missing_captions.append(path)
                else:
                    missing_captions.append(path)

            # Check if item needs processing
            if not cached_data:
                needs_processing = True
            else:
                if process_latents and "image_latent" not in cached_data:
                    needs_processing = True
                if process_text_embeddings and "text_embeddings" not in cached_data:
                    needs_processing = True

            if needs_processing:
                to_process.append(path)

        if not to_process and not missing_captions:
            logger.info("All items are fully processed and cached")
            return

        logger.info(f"Processing {len(to_process)} items, {len(missing_captions)} missing captions")

        # Process items in batches
        with tqdm(total=len(to_process), desc="Processing items") as pbar:
            for i in range(0, len(to_process), batch_size):
                batch_paths = to_process[i:i+batch_size]

                # Periodic CUDA cache clearing
                if torch.cuda.is_available() and i % 100 == 0:
                    torch.cuda.empty_cache()
                    current_memory = torch.cuda.memory_allocated()
                    logger.debug(f"CUDA memory: {current_memory/1024**2:.1f}MB")

                for img_path in batch_paths:
                    try:
                        # Initialize processing data
                        image_latent = None
                        text_embeddings = None
                        metadata = {"path": img_path, "timestamp": time.time()}
                        caption = ""

                        # Get or generate caption
                        caption_path = Path(img_path).with_suffix('.txt')
                        if caption_path.exists():
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        else:
                            # Check cache for caption
                            cached_data = self.cache_manager.get_cached_item(img_path)
                            if cached_data and "metadata" in cached_data:
                                caption = cached_data["metadata"].get("caption", "")
                                if caption:
                                    # Save caption to file
                                    try:
                                        with open(caption_path, 'w', encoding='utf-8') as f:
                                            f.write(caption)
                                    except Exception as e:
                                        logger.warning(f"Failed to write caption file {caption_path}: {e}")    

                        # Apply empty prompt probability
                        if proportion_empty_prompts > 0 and random.random() < proportion_empty_prompts:        
                            caption = ""

                        # Process image latents if needed
                        if process_latents:
                            processed = self._process_image(img_path)
                            if processed:
                                image_latent = processed["image_latent"]
                                metadata.update(processed.get("metadata", {}))

                        # Process text embeddings if needed
                        if process_text_embeddings:
                            embeddings = self.latent_preprocessor.encode_prompt([caption])
                            text_embeddings = embeddings
                            metadata["caption"] = caption

                        # Save to cache
                        if image_latent is not None or text_embeddings is not None:
                            self.cache_manager.save_preprocessed_data(
                                image_latent=image_latent,
                                text_embeddings=text_embeddings,
                                metadata=metadata,
                                file_path=img_path,
                                caption=caption
                            )
                            self.stats.successful += 1

                    except Exception as e:
                        self.stats.failed += 1
                        logger.error(f"Failed to process {img_path}: {e}", exc_info=True)
                    finally:
                        pbar.update(1)

                # Periodic CUDA synchronization
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.synchronize()

        # Final memory report
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            logger.info(f"Final CUDA memory: {final_memory/1024**2:.1f}MB")
            logger.info(f"Memory change: {(final_memory - initial_memory)/1024**2:.1f}MB")

        # Final statistics
        logger.info(f"Preprocessing complete: {self.stats.successful} successful, "
                    f"{self.stats.failed} failed, {len(missing_captions)} missing captions")



    def get_valid_image_paths(self) -> List[str]:
        """Return list of valid image paths found during bucketing."""
        if not hasattr(self, 'valid_image_paths'):
            return []
        return self.valid_image_paths

    
    def _process_image(self, img_path):
        try:
            def gpu_process(tensor):
                tensor = tensor.to(self.device, non_blocking=True)
                vae_dtype = next(self.latent_preprocessor.model.vae.parameters()).dtype
                tensor = tensor.to(dtype=vae_dtype)
                return self.latent_preprocessor.encode_images(tensor)

            img = Image.open(img_path).convert('RGB')
            
            # Use centralized functions
            if not self.validate_image_size(img.size):
                raise ValueError(f"Invalid image size: {img.size}")
                
            resized_img, bucket_idx = self.resize_to_bucket(img)
            
            tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            
            latent_output = self._process_on_gpu(gpu_process, tensor)
            
            metadata = {
                "original_size": img.size,
                "bucket_size": self.buckets[bucket_idx],
                "bucket_index": bucket_idx,
                "path": str(img_path),
                "timestamp": time.time(),
                "device_id": self.device_id if torch.cuda.is_available() else None
            }
            return {"image_latent": latent_output["image_latent"], "metadata": metadata}
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
