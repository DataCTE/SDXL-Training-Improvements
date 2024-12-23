"""High-performance preprocessing pipeline for SDXL training with optimized memory and stream handling."""
import logging
import time
import traceback
from src.core.types import DataType, ModelWeightDtypes
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
from queue import Queue
from src.data.utils.paths import convert_windows_path
from threading import Event, Thread
from typing import Dict, List, Optional, Union, Any, Tuple, TYPE_CHECKING
from PIL import Image
from src.data.config import Config
from src.data.preprocessing.latents import LatentPreprocessor
from src.data.preprocessing.cache_manager import CacheManager
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np

import torch
import torch.cuda
from torch.cuda.amp import autocast

from src.core.memory.tensor import (
    unpin_tensor_,
    tensors_record_stream
)
from .exceptions import (
    PreprocessingError, DataLoadError, PipelineConfigError,
    GPUProcessingError, CacheError, DtypeError, DALIError,
    TensorValidationError, StreamError, MemoryError
)

logger = logging.getLogger(__name__)

@dataclass 
class PipelineStats:
    """Track pipeline processing statistics."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_oom_events: int = 0
    stream_sync_failures: int = 0
    dtype_conversion_errors: int = 0

class PreprocessingPipeline:
    """Optimized parallel preprocessing pipeline with enhanced error handling and monitoring."""
    
    def __init__(
        self,
        config: Config,
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        cache_manager: Optional['CacheManager'] = None,
        is_train: bool = True,
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        device_ids: Optional[List[int]] = None,
        use_pinned_memory: bool = True,
        enable_memory_tracking: bool = True,
        stream_timeout: float = 10.0  # Stream synchronization timeout
    ):
        """Initialize pipeline with enhanced configuration options."""
        self._validate_init_params(
            num_gpu_workers=num_gpu_workers,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            prefetch_factor=prefetch_factor,
            device_ids=device_ids
        )
        self._validate_init_params(
            num_gpu_workers=num_gpu_workers,
            num_cpu_workers=num_cpu_workers,
            num_io_workers=num_io_workers,
            prefetch_factor=prefetch_factor,
            device_ids=device_ids
        )
        
        # Store initialization parameters
        self.config = config
        self.cache_manager = cache_manager
        self.is_train = is_train
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.use_pinned_memory = use_pinned_memory
        self.stream_timeout = stream_timeout
        self.enable_memory_tracking = enable_memory_tracking
        
        # Initialize statistics
        self.stats = PipelineStats()
        
        # Setup memory tracking
        if enable_memory_tracking and torch.cuda.is_available():
            self.memory_high_water_mark = 0
            self.memory_tracker = self._create_memory_tracker()
        else:
            self.memory_tracker = None
            
        # Initialize queues with proper buffer sizes
        queue_size = prefetch_factor * num_gpu_workers
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size) 
        
        # Setup CUDA streams with enhanced error handling and fallback
        self.streams = {}
        if torch.cuda.is_available():
            try:
                for dev_id in self.device_ids:
                    try:
                        # Create streams individually for better error handling
                        compute_stream = torch.cuda.Stream(device=dev_id)
                        transfer_stream = torch.cuda.Stream(device=dev_id)
                        
                        # Verify streams are valid
                        if not compute_stream.query() or not transfer_stream.query():
                            raise StreamError(f"Stream validation failed for device {dev_id}")
                            
                        self.streams[dev_id] = {
                            'compute': compute_stream,
                            'transfer': transfer_stream
                        }
                    except Exception as e:
                        logger.warning(f"Failed to create streams for device {dev_id}: {str(e)}")
                        # Skip this device but continue with others
                        continue
                        
                if not self.streams:
                    logger.warning("No CUDA streams could be created, falling back to synchronous processing")
                    
            except Exception as e:
                logger.warning(f"Stream initialization warning: {str(e)}")
                # Don't raise - allow fallback to synchronous processing
                
        # Initialize component pools
        self._init_worker_pools()
        
        # Setup memory optimizations with proper error handling
        if torch.cuda.is_available():
            try:
                self._setup_memory_optimizations()
            except Exception as e:
                raise MemoryError(
                    "Failed to setup memory optimizations",
                    context={'error': str(e)}
                )
                
        # Initialize pipeline with enhanced error handling
        try:
            self.latent_preprocessor = latent_preprocessor
            if latent_preprocessor:
                self.model = latent_preprocessor.model
                    
            # Create CUDA stream pipeline
            self.cuda_stream = self._create_cuda_pipeline()
            if self.cuda_stream is None:
                logger.warning("CUDA pipeline creation failed, falling back to CPU processing")
                    
        except Exception as e:
            logger.warning(f"Pipeline initialization warning: {str(e)}")
            self.cuda_stream = None
            # Don't raise here - allow fallback to CPU processing

    def precompute_latents(
        self,
        image_paths: List[str],
        captions: List[str],
        latent_preprocessor: Optional[LatentPreprocessor],
        batch_size: int = 1,
        proportion_empty_prompts: float = 0.0
    ) -> None:
        """Precompute and cache latents for a dataset.
        
        Args:
            image_paths: List of image paths to process
            captions: List of captions
            latent_preprocessor: Preprocessor for generating latents
            cache_manager: Optional cache manager
            batch_size: Batch size for processing
            proportion_empty_prompts: Proportion of prompts to leave empty
            is_train: Whether this is for training data
        """
        if not latent_preprocessor or not self.cache_manager:
            logger.info("Skipping latent precomputation - missing preprocessor or cache manager")
            return
            
        # Use training mode setting from initialization
        if not self.is_train:
            logger.info("Skipping latent precomputation for validation data")
            return
            
        logger.info(f"Precomputing latents for {len(image_paths)} images")

        # First check cache index for all images
        to_process = []
        for img_path in image_paths:
            if not self.cache_manager.has_cached_item(img_path):
                to_process.append(img_path)
            else:
                self.stats.cache_hits += 1

        if not to_process:
            logger.info("All latents already cached, skipping precomputation")
            return

        logger.info(f"Processing {len(to_process)} uncached images")
        
        try:
            for i in range(0, len(to_process), batch_size):
                batch_paths = to_process[i:i + batch_size]
                # Get corresponding captions by matching indices
                batch_captions = [captions[image_paths.index(path)] for path in batch_paths]
                
                # Process each image in batch
                for img_path, caption in zip(batch_paths, batch_captions):
                    try:
                            
                        # Process image and cache results
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
                        logger.warning(f"Failed to precompute latents for {img_path}: {str(e)}")
                        self.stats.failed += 1
                        continue
                        
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                    
        except Exception as e:
            logger.error(f"Latent precomputation failed: {str(e)}")
            raise
            
        logger.info(
            f"Completed latent precomputation:\n"
            f"- Processed: {self.stats.successful}/{len(image_paths)}\n"
            f"- Failed: {self.stats.failed_images}\n"
            f"- Cache hits: {self.stats.cache_hits}\n"
            f"- Cache misses: {self.stats.cache_misses}"
        )

    def _process_image(self, img_path: Path) -> Dict[str, Any]:
        """Process single image with optimized memory handling.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Dict containing processed tensors and metadata
            
        Raises:
            PreprocessingError: If processing fails
        """
        try:
            # Load and validate image
            img = Image.open(img_path).convert('RGB')
            
            # Get image metadata
            metadata = {
                "original_size": img.size,
                "path": str(img_path),
                "timestamp": time.time()
            }
            
            # Convert to tensor with memory optimization
            with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                # Process with CUDA if available
                if self.cuda_stream and torch.cuda.is_available():
                    try:
                        tensor = self._process_with_cuda(img)
                    except Exception as e:
                        logger.warning(f"CUDA processing failed, falling back to CPU: {e}")
                        tensor = None
                        
                # CPU fallback
                if not self.cuda_stream or tensor is None:
                    tensor = self._apply_optimized_transforms(
                        torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    )
                
                # Generate latents if preprocessor available
                if self.latent_preprocessor:
                    with torch.no_grad():
                        latent = self.latent_preprocessor.encode_images(tensor.unsqueeze(0))
                else:
                    latent = tensor
                    
                # Record stream for async operations
                if torch.cuda.is_available():
                    tensors_record_stream(torch.cuda.current_stream(), latent)
                    
            return {
                "latent": latent,
                "metadata": metadata
            }
            
        except Exception as e:
            raise PreprocessingError(
                f"Failed to process image {img_path}",
                context={"error": str(e)}
            )

    def _validate_init_params(
        self,
        num_gpu_workers: int,
        num_cpu_workers: int,
        num_io_workers: int,
        prefetch_factor: int,
        device_ids: Optional[List[int]]
    ) -> None:
        """Validate initialization parameters.
        
        Args:
            num_gpu_workers: Number of GPU workers
            num_cpu_workers: Number of CPU workers
            num_io_workers: Number of I/O workers
            prefetch_factor: Prefetch queue depth
            device_ids: List of GPU device IDs
            
        Raises:
            PipelineConfigError: If parameters are invalid
        """
        # Validate worker counts
        if num_gpu_workers < 0:
            raise PipelineConfigError("num_gpu_workers must be non-negative")
        if num_cpu_workers < 0:
            raise PipelineConfigError("num_cpu_workers must be non-negative")
        if num_io_workers < 0:
            raise PipelineConfigError("num_io_workers must be non-negative")
            
        # Validate prefetch factor
        if prefetch_factor < 1:
            raise PipelineConfigError("prefetch_factor must be at least 1")
            
        # Validate device IDs if provided
        if device_ids is not None:
            if not torch.cuda.is_available():
                raise PipelineConfigError("CUDA not available but device_ids provided")
            max_devices = torch.cuda.device_count()
            invalid_ids = [i for i in device_ids if i >= max_devices or i < 0]
            if invalid_ids:
                raise PipelineConfigError(
                    f"Invalid device IDs: {invalid_ids}. "
                    f"Available devices: 0-{max_devices-1}"
                )

    def _init_worker_pools(self):
        """Initialize worker pools for parallel processing."""
        try:
            # Initialize GPU worker pool if CUDA available
            if torch.cuda.is_available() and self.num_gpu_workers > 0:
                self.gpu_pool = ThreadPoolExecutor(
                    max_workers=self.num_gpu_workers,
                    thread_name_prefix="gpu_worker"
                )
            else:
                self.gpu_pool = None
                
            # Initialize CPU worker pool
            self.cpu_pool = ProcessPoolExecutor(
                max_workers=self.num_cpu_workers,
                mp_context=torch.multiprocessing.get_context('spawn')
            )
            
            # Initialize I/O worker pool
            self.io_pool = ThreadPoolExecutor(
                max_workers=self.num_io_workers,
                thread_name_prefix="io_worker"
            )
            
            # Initialize stop event
            self.stop_event = Event()
            
        except Exception as e:
            raise PipelineConfigError(
                "Failed to initialize worker pools",
                {"error": str(e)}
            )

    def _setup_memory_optimizations(self):
        """Setup memory optimizations for preprocessing pipeline."""
        try:
            if torch.cuda.is_available():
                # Enable TF32 for better performance on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable cudnn benchmarking
                torch.backends.cudnn.benchmark = True
                
                # Setup pinned memory if enabled
                if self.use_pinned_memory:
                    torch.cuda.empty_cache()
                    
                # Setup stream synchronization
                self.transfer_stream = torch.cuda.Stream()
                self.compute_stream = torch.cuda.Stream()
                
                # Initialize memory pools
                if hasattr(torch.cuda, 'memory_pool'):
                    torch.cuda.memory_pool().empty_cache()
                    
        except Exception as e:
            raise PipelineConfigError(
                "Failed to setup memory optimizations",
                {"error": str(e)}
            )

    def _create_memory_tracker(self):
        """Create enhanced memory tracking utilities."""
        return {
            'peak_allocated': 0,
            'current_allocated': 0,
            'oom_events': 0,
            'logged_timestamps': [],
            'allocation_history': []
        }

    def _track_memory_usage(self, context: str):
        """Track detailed memory statistics."""
        if self.memory_tracker and torch.cuda.is_available():
            current = torch.cuda.memory_allocated()
            self.memory_tracker['current_allocated'] = current
            self.memory_tracker['peak_allocated'] = max(
                self.memory_tracker['peak_allocated'],
                current
            )
            self.memory_tracker['allocation_history'].append({
                'timestamp': time.time(),
                'context': context,
                'allocated': current,
                'peak': self.memory_tracker['peak_allocated']
            })

    def _create_cuda_pipeline(self) -> Optional[torch.cuda.Stream]:
        """Create CUDA stream pipeline with optimized settings."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping pipeline creation")
            return None

        try:
            # Create dedicated CUDA stream for pipeline
            stream = torch.cuda.Stream()
            return stream
            
        except Exception as e:
            logger.error(f"Failed to create CUDA pipeline: {str(e)}")
            return None

    def _process_with_cuda(self, img: Image.Image) -> torch.Tensor:
        """Process image with CUDA acceleration.
        
        Args:
            img: PIL Image to process
            
        Returns:
            Processed tensor on CUDA device
        """
        try:
            # Convert PIL image to tensor
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            # Move to CUDA and apply transforms
            if torch.cuda.is_available():
                # Convert torch dtype to DataType enum
                if self.latent_preprocessor and hasattr(self.latent_preprocessor.model, 'unet'):
                    unet_dtype = self.latent_preprocessor.model.unet.dtype
                    if unet_dtype == torch.float32:
                        model_dtype = DataType.FLOAT_32
                    elif unet_dtype == torch.float16:
                        model_dtype = DataType.FLOAT_16
                    elif unet_dtype == torch.bfloat16:
                        model_dtype = DataType.BFLOAT_16
                    else:
                        model_dtype = DataType.FLOAT_32
                else:
                    model_dtype = DataType.FLOAT_32
                    
                target_dtype = model_dtype.to_torch_dtype()
                tensor = tensor.cuda(non_blocking=True).to(dtype=target_dtype)
                tensor = self._apply_optimized_transforms(tensor)
                
            return tensor
            
        except Exception as e:
            raise PreprocessingError(
                "CUDA processing failed",
                context={"error": str(e)}
            )

    def _apply_optimized_transforms(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply optimized transforms with enhanced error handling."""
        try:
            # Add batch dimension for channels_last format
            tensor = tensor.unsqueeze(0)
            
            # Now convert to channels last
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            # Apply configured transforms with dtype handling
            if getattr(self.config.transforms, 'normalize', True):
                # Match model dtype and device by checking UNet and VAE
                # Get model dtype by checking UNet
                if self.latent_preprocessor and hasattr(self.latent_preprocessor.model, 'unet'):
                    unet_dtype = self.latent_preprocessor.model.unet.dtype
                    if unet_dtype == torch.float32:
                        model_dtype = DataType.FLOAT_32
                    elif unet_dtype == torch.float16:
                        model_dtype = DataType.FLOAT_16
                    elif unet_dtype == torch.bfloat16:
                        model_dtype = DataType.BFLOAT_16
                    else:
                        model_dtype = DataType.FLOAT_32
                else:
                    model_dtype = DataType.FLOAT_32

                target_dtype = model_dtype.to_torch_dtype()
                target_device = (self.latent_preprocessor.model.device
                               if self.latent_preprocessor 
                               else tensor.device)
                
                # Ensure VAE is on same device
                if self.latent_preprocessor and self.latent_preprocessor.model.vae:
                    self.latent_preprocessor.model.vae.to(device=target_device)
                    
                tensor = tensor.to(device=target_device, dtype=target_dtype)
                
                mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device, dtype=target_dtype)
                std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device, dtype=target_dtype)
                
                # Validate dtype consistency
                if not tensor.dtype == target_dtype:
                    logger.warning(f"Tensor dtype mismatch: {tensor.dtype} vs {target_dtype}")
                    tensor = tensor.to(dtype=target_dtype)
                tensor = tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            
            if getattr(self.config.transforms, 'random_flip', False):
                if torch.rand(1) > 0.5:
                    tensor = tensor.flip(-1)
            
            # Remove batch dimension before returning
            return tensor.squeeze(0)
            
        except Exception as e:
            raise PreprocessingError(
                "Transform application failed",
                context={
                    "tensor_shape": tuple(tensor.shape),
                    "error": str(e)
                }
            )

    def process_batch(
        self,
        batch: Union[List[torch.Tensor], List[Dict[str, Any]]],
        device_id: Optional[int] = None,
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        timeout: Optional[float] = None
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Process batch with optimized memory handling and enhanced error tracking.
        
        Args:
            batch: List of tensors or dictionaries to process
            device_id: Optional device ID for processing
            latent_preprocessor: Optional latent preprocessor
            timeout: Optional timeout for queue operations
            
        Returns:
            Processed batch results
            
        Raises:
            PreprocessingError: If batch processing fails
        """
        try:
            # Track memory if enabled
            if self.enable_memory_tracking:
                self._track_memory_usage("batch_processing")

            # Handle direct tensor processing
            if isinstance(batch[0], torch.Tensor) and device_id is not None and latent_preprocessor is not None:
                # Stack and move to device efficiently
                stacked_batch = torch.stack(batch).to(
                    device=torch.device(device_id),
                    memory_format=torch.channels_last,
                    non_blocking=self.use_pinned_memory
                )

                # Get latents through preprocessor
                with autocast():
                    result = latent_preprocessor.encode_images(stacked_batch)

                self.stats.successful += 1
                return result

            # Handle queue-based processing
            else:
                self.input_queue.put(batch, timeout=timeout)
                result = self.output_queue.get(timeout=timeout)
                
                if isinstance(result, Exception):
                    raise result
                    
                return result

        except Exception as e:
            self.stats.failed += 1
            context = {
                "batch_size": len(batch),
                "error": str(e)
            }
            if device_id is not None:
                context["device_id"] = device_id
            if timeout is not None:
                context["timeout"] = timeout
                
            raise PreprocessingError("Batch processing failed", context=context)

    def __enter__(self):
        """Context manager entry with initialization verification."""
        if not hasattr(self, 'dali_pipeline'):
            raise PipelineConfigError("Pipeline not properly initialized")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with enhanced cleanup."""
        self.stop_event.set()
        
        # Clean input queue
        while not self.input_queue.empty():
            try:
                item = self.input_queue.get_nowait()
                if isinstance(item, torch.Tensor) and self.use_pinned_memory:
                    unpin_tensor_(item)
            except:
                break
                
        # Shutdown workers
        for pool in [self.gpu_pool, self.cpu_pool, self.io_pool]:
            pool.shutdown(wait=True)
            
        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log final statistics
        logger.info(
            f"Pipeline shutdown - Processed: {self.stats.total_processed}, "
            f"Successful: {self.stats.successful}, Failed: {self.stats.failed}"
        )
