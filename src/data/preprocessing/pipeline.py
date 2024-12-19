"""High-performance preprocessing pipeline for SDXL training."""
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Dict, List, Optional, Union, Any
import torch
import torch.cuda
from torch.cuda.amp import autocast
from sdxl_trainer.core.memory.tensor import (
    tensors_to_device_,
    create_stream_context,
    tensors_record_stream,
    torch_gc
)
from sdxl_trainer.core.types import DataType, ModelWeightDtypes
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Orchestrates parallel preprocessing across multiple devices."""
    
    def __init__(
        self,
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        device_ids: Optional[List[int]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_pinned_memory: bool = True,
        compression: Optional[str] = "zstd"
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            num_gpu_workers: Number of GPU workers for encoding
            num_cpu_workers: Number of CPU workers for transforms
            num_io_workers: Number of I/O workers for disk ops
            prefetch_factor: Number of batches to prefetch
            device_ids: List of GPU device IDs to use
            cache_dir: Directory for caching
            use_pinned_memory: Whether to use pinned memory
            compression: Cache compression algorithm
        """
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_pinned_memory = use_pinned_memory
        self.compression = compression

        # Initialize queues
        self.input_queue = Queue(maxsize=prefetch_factor)
        self.output_queue = Queue(maxsize=prefetch_factor)
        self.stop_event = Event()

        # Initialize workers
        self._init_workers()
        
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

    def _prefetch_worker(self):
        """Background worker for prefetching data."""
        try:
            while not self.stop_event.is_set():
                # Get next batch from input queue
                batch = self.input_queue.get()
                if batch is None:
                    break

                # Process batch using DALI
                processed = self.dali_pipeline.run()
                
                # Move to GPU and apply transforms
                futures = []
                for item in processed:
                    future = self.gpu_pool.submit(
                        self._process_item,
                        item
                    )
                    futures.append(future)

                # Get results and put in output queue
                results = [f.result() for f in futures]
                self.output_queue.put(results)

        except Exception as e:
            logger.error(f"Error in prefetch worker: {str(e)}")
            self.stop_event.set()

    def _process_item(self, item: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single item with GPU acceleration."""
        try:
            with autocast(enabled=True):
                # Move to GPU with optimized memory format
                if self.use_pinned_memory:
                    item = item.pin_memory()
                
                with create_stream_context(torch.cuda.current_stream()):
                    tensors_to_device_(
                        item, 
                        device=self.device_ids[0],
                        non_blocking=True
                    )
                    if item.device.type == 'cuda':
                        tensors_record_stream(torch.cuda.current_stream(), item)

                # Custom CUDA stream for async processing
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    # Process tensor
                    processed = self._apply_transforms(item)
                    
                    # Move back to CPU if needed
                    if self.cache_dir:
                        processed = processed.cpu()
                        
                    return {"tensor": processed}

        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            raise

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply transforms using custom CUDA kernels."""
        # TODO: Implement custom CUDA kernels
        return tensor

    def process_batch(self, batch: List[Any]) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of items."""
        self.input_queue.put(batch)
        return self.output_queue.get()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_event.set()
        self.input_queue.put(None)
        self.prefetch_thread.join()
        self.gpu_pool.shutdown()
        self.cpu_pool.shutdown()
        self.io_pool.shutdown()
