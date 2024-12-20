"""High-performance preprocessing pipeline for SDXL training."""
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from .exceptions import (
    PreprocessingError, DataLoadError, PipelineConfigError,
    GPUProcessingError, CacheError, DtypeError, DALIError
)
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING, cast
from src.utils.paths import convert_windows_path

if TYPE_CHECKING:
    from ...data.config import Config
import torch
import torch.cuda
from torch.cuda.amp import autocast
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
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)
from src.core.types import ModelWeightDtypes
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Orchestrates parallel preprocessing across multiple devices."""
    
    def __init__(
        self,
        config: "Config",  # type: ignore
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        device_ids: Optional[List[int]] = None,
        use_pinned_memory: bool = True,
        dtypes: Optional[ModelWeightDtypes] = None
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            config: Training configuration
            num_gpu_workers: Number of GPU workers for encoding
            num_cpu_workers: Number of CPU workers for transforms
            num_io_workers: Number of I/O workers for disk ops
            prefetch_factor: Number of batches to prefetch
            device_ids: List of GPU device IDs to use
            use_pinned_memory: Whether to use pinned memory
        """
        self.config = config
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.use_pinned_memory = use_pinned_memory
        
        # Setup default dtypes if not provided
        self.dtypes = dtypes or ModelWeightDtypes(
            train_dtype=DataType.FLOAT_32,
            fallback_train_dtype=DataType.FLOAT_16,
            unet=DataType.FLOAT_32,
            prior=DataType.FLOAT_32,
            text_encoder=DataType.FLOAT_32,
            text_encoder_2=DataType.FLOAT_32,
            vae=DataType.FLOAT_32,
            effnet_encoder=DataType.FLOAT_32,
            decoder=DataType.FLOAT_32,
            decoder_text_encoder=DataType.FLOAT_32,
            decoder_vqgan=DataType.FLOAT_32,
            lora=DataType.FLOAT_32,
            embedding=DataType.FLOAT_32
        )
        
        # Initialize cache paths and manager
        self.cache_dir = Path(convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager if enabled
        self.cache_manager = None
        if config.global_config.cache.use_cache:
            from .cache_manager import CacheManager
            self.cache_manager = CacheManager(
                cache_dir=self.cache_dir,
                compression=config.global_config.cache.compression if hasattr(config.global_config.cache, 'compression') else 'zstd'
            )
            
        # Initialize config reference for _save_processed_batch
        self.config = config

        # Initialize queues
        self.input_queue = Queue(maxsize=prefetch_factor)
        self.output_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()

        # Initialize workers
        self._init_workers()
        
        # Setup memory optimizations
        if torch.cuda.is_available():
            setup_memory_optimizations(
                model=None,
                config=config,
                device=torch.device("cuda"),
                batch_size=32  # DALI pipeline batch size
            )
        
        # Setup DALI pipeline
        self.dali_pipeline = self._create_dali_pipeline()

    def _init_workers(self):
        """Initialize worker pools."""
        # GPU workers for encoding
        self.gpu_pool = ProcessPoolExecutor(
            max_workers=self.num_gpu_workers,
            mp_context=torch.multiprocessing.get_context('spawn')
        )
        
        # CPU workers for transforms
        self.cpu_pool = ThreadPoolExecutor(
            max_workers=self.num_cpu_workers
        )
        
        # I/O workers for disk operations
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.num_io_workers
        )

        # Start prefetch thread
        self.prefetch_thread = Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()

    def _create_dali_pipeline(self) -> Pipeline:
        """Create DALI pipeline for image loading."""
        pipe = Pipeline(
            batch_size=32,
            num_threads=self.num_cpu_workers,
            device_id=self.device_ids[0]
        )

        with pipe:
            # Define operations
            images = fn.readers.file(
                name="Reader",
                pad_last_batch=True,
                random_shuffle=True
            )
            decoded = fn.decoders.image(
                images,
                device="mixed"
            )
            normalized = fn.crop_mirror_normalize(
                decoded,
                dtype=dali.types.FLOAT,
                mean=[0.5 * 255] * 3,
                std=[0.5 * 255] * 3
            )
            pipe.set_outputs(normalized)

        return pipe

    def _prefetch_worker(self) -> None:
        """Background worker for prefetching data.
        
        Continuously pulls batches from input queue, processes them
        through DALI pipeline and GPU transforms, then puts results
        in output queue. Runs in separate thread.
        
        Raises:
            DALIError: If DALI pipeline processing fails
            PreprocessingError: For other processing errors
        """
        try:
            while not self.stop_event.is_set():
                try:
                    # Get next batch from input queue with timeout
                    batch = self.input_queue.get(timeout=1.0)
                    if batch is None:
                        break

                    # Validate batch
                    if not isinstance(batch, (list, tuple)):
                        raise DataLoadError(f"Expected list or tuple, got {type(batch)}")

                    # Process batch using DALI
                    try:
                        processed = self.dali_pipeline.run()
                    except Exception as e:
                        raise DALIError(f"DALI pipeline failed: {str(e)}") from e
            
                # Move to GPU and apply transforms with proper dtype
                futures = []
                for item in processed:
                    # Convert DALI output to torch tensor with target dtype
                    tensor = torch.as_tensor(item, dtype=self.dtypes.train_dtype.to_torch_dtype())
                    future = self.gpu_pool.submit(
                        self._process_item,
                        tensor
                    )
                    futures.append(future)

                # Get results and put in output queue
                results = [f.result() for f in futures]
                self.output_queue.put(results)

        except Exception as e:
            logger.error(f"Error in prefetch worker: {str(e)}")
            self.stop_event.set()

    def _process_item(self, item: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single item with GPU acceleration and memory optimizations.
        
        Args:
            item: Input tensor to process
            
        Returns:
            Dictionary containing processed tensor
            
        Raises:
            GPUProcessingError: If GPU processing fails
            DtypeError: If dtype conversion fails
            PreprocessingError: For other processing errors
        """
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(item)}")
            
        try:
            # Validate input tensor
            if item.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got {item.dim()}D")
                
            # Setup memory optimizations
            if torch.cuda.is_available():
                try:
                    verify_memory_optimizations(
                        model=None,
                        config=self.config,
                        device=torch.device("cuda"),
                        logger=logger
                    )
                except Exception as e:
                    raise GPUProcessingError(f"Failed to setup memory optimizations: {str(e)}") from e
            
            # Use multiple streams for pipelining
            compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            
            # Pre-allocate and use pinned memory buffer
            if self.use_pinned_memory and torch.cuda.is_available():
                if not hasattr(item, 'is_pinned') or not item.is_pinned():
                    pin_tensor_(item)
            
            with autocast(enabled=True):
                # Optimize memory format
                item = item.to(memory_format=torch.channels_last, non_blocking=True)
                
                # Use transfer stream for device moves
                if transfer_stream is not None:
                    with torch.cuda.stream(transfer_stream):
                        # Only transfer if device mismatch
                        if not device_equals(item.device, torch.device(self.device_ids[0])):
                            tensors_to_device_(
                                item,
                                device=self.device_ids[0],
                                non_blocking=True
                            )
                        # Record stream for proper synchronization
                        if item.device.type == 'cuda':
                            tensors_record_stream(transfer_stream, item)
                            transfer_stream.synchronize()

                # Use compute stream for processing
                if compute_stream is not None:
                    compute_stream.wait_stream(transfer_stream)
                with torch.cuda.stream(compute_stream):
                    # Process tensor with memory optimizations
                    processed = self._apply_transforms(item)
                
                    # Replace with optimized tensor if needed
                    if hasattr(processed, 'data_ptr'):
                        replace_tensors_(item, processed)
                    
                    # Move back to CPU if needed, with proper cleanup
                    if self.cache_dir:
                        with create_stream_context(compute_stream):
                            # Only transfer if device mismatch
                            if not device_equals(processed.device, torch.device('cpu')):
                                processed = processed.cpu()
                                torch_gc()
                            
                            # Pin memory for faster transfers if enabled
                            if self.use_pinned_memory:
                                pin_tensor_(processed)
                            
                    return {"tensor": processed}

        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            raise

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply transforms using custom CUDA kernels.
        
        Args:
            tensor: Input tensor to transform
            
        Returns:
            Transformed tensor
            
        Raises:
            DtypeError: If dtype conversion fails
            GPUProcessingError: If GPU operations fail
            ValueError: If tensor format is invalid
        """
        # Validate input
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
            
        # Ensure tensor is in correct format
        if tensor.dim() != 4:  # [B, C, H, W]
            raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")
            
        # Convert to target dtype before processing
        try:
            tensor = tensor.to(dtype=self.dtypes.train_dtype.to_torch_dtype())
        except Exception as e:
            raise DtypeError(f"Failed to convert tensor to {self.dtypes.train_dtype}: {str(e)}") from e
            
        # Apply transforms in compute stream
        with torch.cuda.stream(torch.cuda.current_stream()):
            # Convert to channels last for better performance
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            # Normalize if needed (assuming input is in [0,1])
            if tensor.max() <= 1.0:
                tensor = tensor * 2.0 - 1.0
                
            # Apply custom transforms based on config
            if hasattr(self.config, 'transforms'):
                if self.config.transforms.get('random_flip', False):
                    if torch.rand(1).item() < 0.5:
                        tensor = tensor.flip(-1)  # Horizontal flip
                        
                if self.config.transforms.get('normalize', True):
                    # Apply mean/std normalization
                    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
                    tensor = (tensor - mean) / std
                    
            # Ensure compute stream is synchronized
            if tensor.device.type == 'cuda':
                torch.cuda.current_stream().synchronize()
                
        return tensor

    def process_batch(
        self,
        batch: List[Union[torch.Tensor, Dict[str, Any]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of items."""
        self.input_queue.put(batch)
        return self.output_queue.get()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit with cleanup."""
        self.stop_event.set()
        self.input_queue.put(None)
        self.prefetch_thread.join()
        
        # Cleanup pinned memory
        if self.use_pinned_memory:
            for item in list(self.input_queue.queue):
                if isinstance(item, torch.Tensor):
                    unpin_tensor_(item)
                    
        self.gpu_pool.shutdown()
        self.cpu_pool.shutdown()
        self.io_pool.shutdown()
