"""High-performance preprocessing pipeline for SDXL training with optimized memory and stream handling."""
import logging
import time
import traceback
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

import torch
import torch.cuda
from torch.cuda.amp import autocast
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline

from src.core.memory.tensor import (
    unpin_tensor_
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
        
        self.config = config
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
                
        # Initialize encoders and pipeline with enhanced error handling
        try:
            if latent_preprocessor:
                self.model = latent_preprocessor.model
                    
            # Create DALI pipeline with fallback
            self.dali_pipeline = self._create_optimized_dali_pipeline()
            if self.dali_pipeline is None:
                logger.warning("DALI pipeline creation failed, falling back to CPU processing")
                    
        except Exception as e:
            logger.warning(f"Pipeline initialization warning: {str(e)}")
            self.dali_pipeline = None
            # Don't raise here - allow fallback to CPU processing

    def precompute_latents(
        self,
        image_paths: List[str],
        captions: List[str],
        latent_preprocessor: Optional[LatentPreprocessor],
        cache_manager: Optional[CacheManager],
        batch_size: int = 1,
        proportion_empty_prompts: float = 0.0,
        is_train: bool = True
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
        if not latent_preprocessor or not cache_manager:
            logger.info("Skipping latent precomputation - missing preprocessor or cache manager")
            return
            
        logger.info(f"Precomputing latents for {len(image_paths)} images")
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_captions = captions[i:i + batch_size]
                
                # Process each image in batch
                for img_path, caption in zip(batch_paths, batch_captions):
                    try:
                        # Skip if already cached
                        if cache_manager.has_cached_item(img_path):
                            self.stats.cache_hits += 1
                            continue
                            
                        # Process image and cache results
                        processed = self._process_image(img_path)
                        if processed:
                            cache_manager.save_preprocessed_data(
                                latent_data=processed["latent"],
                                text_embeddings=processed.get("text_embeddings"),
                                metadata=processed.get("metadata", {}),
                                file_path=img_path
                            )
                            self.stats.cache_misses += 1
                            self.stats.successful += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to precompute latents for {img_path}: {str(e)}")
                        self.stats.failed_images += 1
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

    def _create_optimized_dali_pipeline(self) -> Optional[Pipeline]:
        """Create DALI pipeline with optimized settings and proper error handling."""
        if not torch.cuda.is_available():
            logger.warning("cuda not available, skipping DALI pipeline creation")
            return None

        try:
            # Create pipeline with device verification
            device_id = self.device_ids[0] if self.device_ids else 0
            if device_id >= torch.cuda.device_count():
                raise DALIError(f"Invalid device ID {device_id}")

            pipe = Pipeline(
                batch_size=32,
                num_threads=self.num_cpu_workers,
                device_id=device_id,
                prefetch_queue_depth=self.prefetch_factor,
                enable_memory_stats=self.enable_memory_tracking
            )

            # Setup pipeline operations within context
            with pipe:
                # Enhanced error handling for readers with stream synchronization
                with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                    try:
                        images = fn.readers.file(
                            name="Reader",
                            pad_last_batch=True,
                            random_shuffle=True,
                            prefetch_queue_depth=self.prefetch_factor
                        )
                    except Exception as e:
                        raise DALIError(
                            "Failed to initialize DALI file reader",
                            context={'error': str(e)}
                        )

                    # Optimized decode settings with memory padding
                    try:
                        decoded = fn.decoders.image(
                            images,
                            device="mixed",
                            output_type=dali.types.RGB,
                            hybrid_huffman_threshold=100000,
                            host_memory_padding=512,  # Increased padding
                            device_memory_padding=512
                        )
                    except Exception as e:
                        raise DALIError(
                            "Failed to initialize DALI image decoder",
                            context={'error': str(e)}
                        )

                    # Enhanced normalization with better precision
                    try:
                        normalized = fn.crop_mirror_normalize(
                            decoded,
                            dtype=dali.types.FLOAT,
                            mean=[0.5 * 255] * 3,
                            std=[0.5 * 255] * 3,
                            output_layout="CHW",
                            pad_output=False
                        )
                    except Exception as e:
                        raise DALIError(
                            "Failed to initialize DALI normalization",
                            context={'error': str(e)}
                        )

                    # Set pipeline output with proper expansion
                    pipe.set_outputs(normalized)

            # Build and verify pipeline
            try:
                pipe.build()
                return pipe
            except Exception as e:
                logger.error(f"Failed to build DALI pipeline: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to create DALI pipeline: {str(e)}")
            return None

    def _apply_optimized_transforms(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply optimized transforms with enhanced error handling."""
        try:
            # Convert to channels last for better performance
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            # Apply configured transforms
            if self.config.transforms.get('normalize', True):
                mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device)
                std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device)
                tensor = tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
            
            if self.config.transforms.get('random_flip', False):
                if torch.rand(1) > 0.5:
                    tensor = tensor.flip(-1)
            
            return tensor
            
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
