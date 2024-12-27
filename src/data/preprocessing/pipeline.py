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
        # Add embedding processor reference
        self.embedding_processor = latent_preprocessor.embedding_processor if latent_preprocessor else None
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
        
        # Initialize thread-local storage for streams
        self._streams = {}
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
            size: Tuple of (width, height)  # Note: PIL Image.size returns (width, height)
            min_size: Optional minimum dimensions
            max_size: Optional maximum dimensions
            
        Returns:
            bool indicating if size is valid
        """
        min_size = min_size or self.config.global_config.image.min_size
        max_size = max_size or self.config.global_config.image.max_size
        
        w, h = size  # PIL returns (width, height)
        min_h, min_w = min_size
        max_h, max_w = max_size
        
        # Calculate total pixels
        total_pixels = w * h
        max_pixels = self.config.global_config.image.max_dim
        
        # Check total pixels first
        if total_pixels > max_pixels:
            logger.warning(
                f"Image too large: {w}x{h} = {total_pixels:,} pixels exceeds maximum {max_pixels:,} pixels"
            )
            return False
            
        # Check minimum dimensions
        if w < min_w or h < min_h:
            logger.warning(
                f"Image too small: {w}x{h} is smaller than minimum dimensions {min_w}x{min_h}"
            )
            return False
        
        # Check maximum dimensions
        if w > max_w or h > max_h:
            logger.warning(
                f"Image too large: {w}x{h} exceeds maximum dimensions {max_w}x{max_h}"
            )
            return False
                
        # Check aspect ratio
        aspect = w / h
        max_aspect = self.config.global_config.image.max_aspect_ratio
        min_aspect = 1.0 / max_aspect
        
        if aspect < min_aspect or aspect > max_aspect:
            logger.warning(
                f"Invalid aspect ratio: {aspect:.2f} (allowed range: {min_aspect:.2f}-{max_aspect:.2f})"
            )
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

    def _process_text_embeddings(
        self,
        image_paths: List[str],
        batch_size: int = 1,
        proportion_empty_prompts: float = 0.0
    ) -> None:
        """Process text embeddings with improved handling."""
        text_embeddings_dir = Path(self.cache_manager.cache_dir) / "text_latents"
        cached_text_files = set(p.stem for p in text_embeddings_dir.glob("*.pt"))
        
        missing_embeddings = []
        for img_path in image_paths:
            img_path = Path(img_path)
            txt_path = img_path.parent / f"{img_path.stem}.txt"
            
            if txt_path.exists() and img_path.stem not in cached_text_files:
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                        
                    if caption and random.random() >= proportion_empty_prompts:
                        missing_embeddings.append({
                            "caption": caption,
                            "img_id": img_path.stem,
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
                            with torch.amp.autocast('cuda'):
                                embeddings = self.latent_preprocessor.encode_prompt([item["caption"]])
                            
                            self.cache_manager.save_preprocessed_data(
                                image_latent=None,
                                text_latent={
                                    "embeddings": embeddings["embeddings"],
                                    "caption": item["caption"]
                                },
                                metadata={
                                    "image_path": item["img_path"],
                                    "text_path": item["txt_path"],
                                    "timestamp": time.time()
                                },
                                file_path=item["img_id"]
                            )
                            self.stats.successful += 1
                            
                        except Exception as e:
                            self.stats.failed += 1
                            logger.error(f"Failed to process caption from {item['txt_path']}: {e}")
                        finally:
                            pbar.update(1)
                            
                    if torch.cuda.is_available() and i % 100 == 0:
                        torch.cuda.empty_cache()
        else:
            logger.info("All text embeddings already cached")
        logger.info(f"Precomputation complete. Successful: {self.stats.successful}, Failed: {self.stats.failed}")


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

        # Setup cache directories
        image_latents_dir = self.cache_manager.image_latents_dir
        text_latents_dir = self.cache_manager.text_latents_dir
        
        # Get cached file lists
        cached_image_files = set(p.stem for p in image_latents_dir.glob("*.pt"))
        cached_text_files = set(p.stem for p in text_latents_dir.glob("*.pt"))
        
        # Build verified index
        new_index = {"files": {}}
        for img_path in image_paths:
            base_name = Path(img_path).stem
            file_info = {}
            
            if base_name in cached_image_files:
                file_info["latent_path"] = str(image_latents_dir / f"{base_name}.pt")
                
            if base_name in cached_text_files:
                file_info["text_path"] = str(text_latents_dir / f"{base_name}.pt")
                
            if file_info:
                new_index["files"][str(img_path)] = file_info

        # Update cache index
        self.cache_manager.cache_index = new_index
        self.cache_manager._save_index()
        
        # Process image latents
        if process_latents:
            missing_latents = [
                img_path for img_path in image_paths
                if Path(img_path).stem not in cached_image_files
            ]

            if missing_latents:
                logger.info(f"Processing {len(missing_latents)} missing image latents")
                save_interval = max(100, len(missing_latents) // 20)
                last_save = 0
                
                with tqdm(total=len(missing_latents), desc="Processing image latents") as pbar:
                    for i in range(0, len(missing_latents), batch_size):
                        batch_paths = missing_latents[i:i+batch_size]
                        
                        for img_path in batch_paths:
                            try:
                                with torch.amp.autocast('cuda'):
                                    processed = self._process_image(img_path)
                                    if processed:
                                        metadata = {
                                            **(processed.get("metadata", {}) if isinstance(processed, dict) else {}),
                                            "path": img_path,
                                            "timestamp": time.time()
                                        }
                                        
                                        image_latent = processed["image_latent"] if isinstance(processed, dict) else processed
                                        self.cache_manager.save_preprocessed_data(
                                            image_latent=image_latent,
                                            metadata=metadata,
                                            file_path=img_path
                                        )
                                        self.stats.successful += 1
                            except Exception as e:
                                self.stats.failed += 1
                                if not isinstance(e, ValueError):
                                    logger.error(f"Failed to process image {img_path}: {e}")
                            finally:
                                pbar.update(1)
                        
                        # Periodic saves and cleanup
                        if i - last_save >= save_interval:
                            self.cache_manager._save_index()
                            last_save = i
                        if torch.cuda.is_available() and i % 100 == 0:
                            torch.cuda.empty_cache()

                self.cache_manager._save_index()
            else:
                logger.info("All image latents already cached")

        # Process text embeddings
        if process_text_embeddings:
            missing_embeddings = []
            for img_path in image_paths:
                img_path = Path(img_path)
                txt_path = img_path.parent / f"{img_path.stem}.txt"
                
                if txt_path.exists() and img_path.stem not in cached_text_files:
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                            
                        if caption and random.random() >= proportion_empty_prompts:
                            missing_embeddings.append({
                                "caption": caption,
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
                                with torch.cuda.amp.autocast():
                                    text_embeddings = self.latent_preprocessor.encode_prompt([item["caption"]])
                                    
                                    self.cache_manager.save_preprocessed_data(
                                        image_latent=None,
                                        text_latent={
                                            "embeddings": text_embeddings.get("embeddings", {}),
                                            "caption": item["caption"]
                                        },
                                        metadata={
                                            "image_path": item["img_path"],
                                            "text_path": item["txt_path"],
                                            "timestamp": time.time()
                                        },
                                        file_path=item["img_path"]
                                    )
                                    self.stats.successful += 1
                            except Exception as e:
                                self.stats.failed += 1
                                logger.error(f"Failed to process caption from {item['txt_path']}: {e}")
                            finally:
                                pbar.update(1)
                                
                        if torch.cuda.is_available() and i % 100 == 0:
                            torch.cuda.empty_cache()
            else:
                logger.info("All text embeddings already cached")

        logger.info(f"Precomputation complete. Successful: {self.stats.successful}, Failed: {self.stats.failed}")
        
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
        
    def _validate_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """Validate tensor and handle invalid values."""
        if tensor is None:
            raise ValueError(f"Tensor {name} is None")
            
        if torch.isnan(tensor).any():
            logger.warning(f"Found NaN values in {name}, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0)
            
        if torch.isinf(tensor).any():
            logger.warning(f"Found infinite values in {name}, clipping")
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
            
        return tensor

    def _process_image(self, img_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process single image with enhanced error handling."""
        try:
            img = Image.open(img_path).convert('RGB')
            if not self.validate_image_size(img.size):
                logger.warning(f"Image {img_path} failed size validation")
                return None

            resized_img, bucket_idx = self.resize_to_bucket(img)
            
            tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            # Validate input tensor
            tensor = self._validate_tensor(tensor, "input")
            
            with torch.cuda.amp.autocast():
                with self.track_memory_usage("vae_encoding"):
                    vae_output = self._process_on_gpu(
                        self.latent_preprocessor.encode_images,
                        tensor
                    )
                    
                if vae_output is None:
                    raise ValueError(f"VAE encoding failed for {img_path}")
                    
                latents = self._validate_tensor(vae_output["image_latent"], "latents")
                uncond_latents = self._validate_tensor(
                    vae_output.get("uncond_latents"), 
                    "uncond_latents"
                )
                
                metadata = {
                    "original_size": img.size,
                    "bucket_size": self.buckets[bucket_idx],
                    "bucket_index": bucket_idx,
                    "path": str(img_path),
                    "timestamp": time.time(),
                    "scaling_factor": vae_output.get("scaling_factor", 0.18215),
                    "latent_shape": vae_output.get("latent_shape"),
                    "input_shape": vae_output.get("input_shape"),
                    "vae_stats": vae_output.get("stats", {})
                }
                
                result = {"image_latent": latents, "metadata": metadata}
                if uncond_latents is not None:
                    result["uncond_latents"] = uncond_latents
                    
                self.stats.successful += 1
                return result
                
        except Exception as e:
            self.stats.failed += 1
            logger.error(f"Failed to process {img_path}: {str(e)}")
            self.performance_stats['errors'].append({
                'path': str(img_path),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            })
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

    def _get_stream(self):
        """Get a CUDA stream for the current thread."""
        import threading
        thread_id = threading.get_ident()
        if thread_id not in self._streams and torch.cuda.is_available():
            self._streams[thread_id] = torch.cuda.Stream()
        return self._streams.get(thread_id)

    def _process_on_gpu(self, func, *args, **kwargs):
        """Execute function on GPU with proper stream management."""
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
            
        try:
            with torch.cuda.device(self.device_id):
                stream = self._get_stream()
                with create_stream_context(stream):
                    result = func(*args, **kwargs)
                    if stream:
                        stream.synchronize()
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
