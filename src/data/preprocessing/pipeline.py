"""High-performance preprocessing pipeline for SDXL training with optimized memory and stream handling."""
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
from src.data.config import Config
from src.data.preprocessing.latents import LatentPreprocessor
from dataclasses import dataclass

import torch
import torch.cuda
from torch.cuda.amp import autocast
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline

from src.core.types import DataType, ModelWeightDtypes
from src.core.memory.tensor import (
    tensors_to_device_,
    create_stream_context,
    tensors_record_stream,
    torch_gc,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_
)
from src.models.encoders.vae import VAEEncoder
from src.models.encoders.clip import encode_clip
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)
from src.data.utils.paths import convert_windows_path

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
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        device_ids: Optional[List[int]] = None,
        use_pinned_memory: bool = True,
        dtypes: Optional[ModelWeightDtypes] = None,
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
        
        # Setup CUDA streams with error checking
        if torch.cuda.is_available():
            try:
                self.streams = {
                    dev_id: {
                        'compute': torch.cuda.Stream(device=dev_id),
                        'transfer': torch.cuda.Stream(device=dev_id)
                    }
                    for dev_id in self.device_ids
                }
            except Exception as e:
                raise StreamError(
                    "Failed to initialize CUDA streams",
                    context={
                        'device_ids': self.device_ids,
                        'error': str(e)
                    }
                )
                
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
                
        # Initialize encoders and pipeline
        try:
            if latent_preprocessor:
                self.vae_encoder = latent_preprocessor.vae_encoder
                self.text_encoder_one = latent_preprocessor.text_encoder_one
                self.text_encoder_two = latent_preprocessor.text_encoder_two
                self.tokenizer_one = latent_preprocessor.tokenizer_one
                self.tokenizer_two = latent_preprocessor.tokenizer_two
            
            self.dali_pipeline = self._create_optimized_dali_pipeline()
        except Exception as e:
            raise DALIError(
                "Failed to create pipeline",
                context={'error': str(e)}
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

    def _create_optimized_dali_pipeline(self) -> Pipeline:
        """Create DALI pipeline with optimized settings."""
        pipe = Pipeline(
            batch_size=32,
            num_threads=self.num_cpu_workers,
            device_id=self.device_ids[0],
            prefetch_queue_depth=self.prefetch_factor,
            enable_memory_stats=self.enable_memory_tracking
        )

        with pipe:
            # Enhanced error handling for readers
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

            # Optimized decode settings
            try:
                decoded = fn.decoders.image(
                    images,
                    device="mixed",
                    output_type=dali.types.RGB,
                    hybrid_huffman_threshold=100000,
                    host_memory_padding=256,
                    device_memory_padding=256
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

            pipe.set_outputs(normalized)

        return pipe

    @torch.no_grad()
    def _process_tensor_batch(
        self,
        tensor: torch.Tensor,
        device_id: int,
        text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Process tensor batch with optimized memory handling and enhanced error checking."""
        try:
            # Validate tensor before processing
            self._validate_input_tensor(tensor)
            
            # Track memory usage
            if self.enable_memory_tracking:
                self._track_memory_usage('pre_process')
            
            # Get device streams
            streams = self.streams[device_id]
            compute_stream = streams['compute']
            transfer_stream = streams['transfer']
            
            # Pin memory if configured
            if self.use_pinned_memory:
                pin_tensor_(tensor)
            
            try:
                # Transfer to device with proper stream handling
                with create_stream_context() as (transfer_ctx,):
                    with torch.cuda.stream(transfer_stream):
                        if not device_equals(tensor.device, torch.device(device_id)):
                            tensors_to_device_(tensor, device_id, non_blocking=True)
                        tensors_record_stream(transfer_stream, tensor)
                        
                # Process with compute stream
                with create_stream_context() as (compute_ctx,):
                    with torch.cuda.stream(compute_stream):
                        # Wait for transfer
                        compute_stream.wait_stream(transfer_stream)
                        
                        # Process with optimized encoders
                        with autocast():
                            # Generate VAE latents
                            latents = self.vae_encoder.encode(tensor, return_dict=False)
                            
                            # Generate text embeddings if text provided
                            if text and hasattr(self, 'text_encoder_one'):
                                # Tokenize
                                tokens_1 = self.tokenizer_one(
                                    text,
                                    padding="max_length",
                                    max_length=self.tokenizer_one.model_max_length,
                                    truncation=True,
                                    return_tensors="pt"
                                ).input_ids.to(device_id)
                                
                                tokens_2 = self.tokenizer_two(
                                    text,
                                    padding="max_length",
                                    max_length=self.tokenizer_two.model_max_length,
                                    truncation=True,
                                    return_tensors="pt"
                                ).input_ids.to(device_id)
                                
                                # Encode text
                                text_embeddings_1, _ = encode_clip(
                                    text_encoder=self.text_encoder_one,
                                    tokens=tokens_1,
                                    default_layer=-2,
                                    layer_skip=0,
                                    use_attention_mask=False,
                                    add_layer_norm=False
                                )
                                
                                text_embeddings_2, pooled_embeddings = encode_clip(
                                    text_encoder=self.text_encoder_two,
                                    tokens=tokens_2,
                                    default_layer=-2,
                                    layer_skip=0,
                                    add_pooled_output=True,
                                    use_attention_mask=False,
                                    add_layer_norm=False
                                )
                                
                                processed = {
                                    "latents": latents,
                                    "text_embeddings_1": text_embeddings_1,
                                    "text_embeddings_2": text_embeddings_2,
                                    "pooled_embeddings": pooled_embeddings
                                }
                            else:
                                processed = {"latents": latents}
                            
                        # Ensure compute is done
                        compute_stream.synchronize()
                        
                # Update statistics
                self.stats.successful += 1
                
                return {
                    "tensor": processed,
                    "metadata": {
                        "device_id": device_id,
                        "shape": tuple(processed.shape),
                        "dtype": str(processed.dtype),
                        "memory_allocated": torch.cuda.memory_allocated(device_id)
                    }
                }
                
            except Exception as e:
                self.stats.failed += 1
                raise PreprocessingError(
                    "Failed to process tensor batch",
                    context={
                        "device_id": device_id,
                        "tensor_shape": tuple(tensor.shape),
                        "error": str(e)
                    }
                )
                
            finally:
                # Cleanup
                if self.use_pinned_memory:
                    unpin_tensor_(tensor)
                if self.enable_memory_tracking:
                    self._track_memory_usage('post_process')
                    
        except Exception as e:
            raise PreprocessingError(
                "Tensor batch processing failed",
                context={
                    "error": str(e),
                    "device_id": device_id
                }
            )

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
        batch: List[Union[torch.Tensor, Dict[str, Any]]],
        timeout: Optional[float] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Process batch with timeout and enhanced error handling."""
        try:
            self.input_queue.put(batch, timeout=timeout)
            result = self.output_queue.get(timeout=timeout)
            
            if isinstance(result, Exception):
                raise result
                
            return result
            
        except Exception as e:
            raise PreprocessingError(
                "Batch processing failed",
                context={
                    "batch_size": len(batch),
                    "timeout": timeout,
                    "error": str(e)
                }
            )

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
