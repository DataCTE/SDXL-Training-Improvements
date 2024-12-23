"""High-performance cache management with optimized tensor and memory handling."""
import multiprocessing as mp
import traceback
from src.core.types import DataType, ModelWeightDtypes
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from ..utils.paths import convert_windows_path
import hashlib
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
from src.data.config import Config

from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_,
    torch_sync
)
import logging
from contextlib import nullcontext

from src.core.logging.logging import setup_logging
from ..utils.paths import convert_windows_path

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Enhanced statistics for cache operations."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    corrupted_items: int = 0
    validation_errors: int = 0
    memory_errors: int = 0
    io_errors: int = 0
    gpu_oom_events: int = 0

class CacheManager:
    """Manages high-throughput caching with optimized memory handling."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        num_proc: Optional[int] = None,
        chunk_size: int = 1000,
        compression: Optional[str] = "zstd",
        verify_hashes: bool = True,
        max_memory_usage: float = 0.8,
        enable_memory_tracking: bool = True,
        stream_buffer_size: int = 1024 * 1024,  # 1MB stream buffer
        max_chunk_memory: float = 0.2  # Max memory per chunk as fraction of total
    ):
        """Initialize cache manager with enhanced memory management.
        
        Args:
            cache_dir: Directory for cached files
            num_proc: Number of processes (default: CPU count)
            chunk_size: Number of items per cache chunk
            compression: Compression algorithm (None, 'zstd', 'gzip')
            verify_hashes: Whether to verify content hashes
            max_memory_usage: Maximum fraction of GPU memory to use
            enable_memory_tracking: Whether to track memory usage
        """
        # Setup base configuration
        self.cache_dir = Path(convert_windows_path(cache_dir, make_absolute=True))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_proc = num_proc or mp.cpu_count()
        self.chunk_size = chunk_size
        self.compression = compression
        self.verify_hashes = verify_hashes
        self.max_memory_usage = max_memory_usage
        self.enable_memory_tracking = enable_memory_tracking
        
        # Initialize statistics
        self.stats = CacheStats()
        self.memory_stats = {
            'peak_allocated': 0,
            'total_allocated': 0,
            'num_allocations': 0,
            'oom_events': 0
        }
        
        # Setup cache paths
        self.text_dir = self.cache_dir / "text"
        self.image_dir = self.cache_dir / "image"
        
        # Create cache directories
        for directory in [self.text_dir, self.image_dir]:
            directory.mkdir(exist_ok=True)
        
        # Configure streaming and chunking
        self.stream_buffer_size = stream_buffer_size
        self.max_chunk_memory = max_chunk_memory
        
        # Setup worker pools with proper resource limits
        self.image_pool = ProcessPoolExecutor(
            max_workers=self.num_proc,
            mp_context=mp.get_context('spawn')
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.num_proc * 2,
            thread_name_prefix="cache_io"
        )
        
        # Initialize streaming buffers
        if torch.cuda.is_available():
            self.pinned_buffer = torch.empty(
                (stream_buffer_size,),
                dtype=torch.uint8,
                pin_memory=True
            )
        
        # Load cache index
        self.index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk or create new one."""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            return {"files": {}, "chunks": {}}
        except Exception as e:
            logger.error(f"Failed to load cache index: {str(e)}")
            return {"files": {}, "chunks": {}}
            
    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {str(e)}")

    def _init_memory_tracking(self):
        """Initialize memory tracking utilities."""
        if not hasattr(self, 'memory_stats'):
            self.memory_stats = {
                'peak_allocated': 0,
                'total_allocated': 0,
                'num_allocations': 0,
                'oom_events': 0
            }

    def _track_memory(self, context: str):
        """Track memory usage with proper cleanup."""
        if self.enable_memory_tracking and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                self.memory_stats['peak_allocated'] = max(
                    self.memory_stats['peak_allocated'],
                    allocated
                )
                self.memory_stats['total_allocated'] += allocated
                self.memory_stats['num_allocations'] += 1
                
                # Log if approaching limit
                if allocated > self.max_memory_usage * torch.cuda.get_device_properties(0).total_memory:
                    logger.warning(f"High memory usage in {context}: {allocated / 1e9:.2f}GB")
                    
            except Exception as e:
                logger.error(f"Memory tracking error in {context}: {str(e)}")

    def save_preprocessed_data(
        self,
        latent_data: Optional[Dict[str, torch.Tensor]],
        text_embeddings: Optional[Dict[str, torch.Tensor]],
        metadata: Dict,
        file_path: Union[str, Path]
    ) -> bool:
        """Save preprocessed data with optimized memory handling.
        
        Args:
            latent_data: Dictionary containing latent tensors
            text_embeddings: Dictionary containing text embeddings
            metadata: Associated metadata
            file_path: Path to original file
            
        Returns:
            bool indicating success
        """
        try:
            # Validate inputs - allow either latents or text embeddings to be None
            if latent_data is None and text_embeddings is None:
                logger.error(f"Both latent_data and text_embeddings are None for {file_path}")
                return False

            # Track memory before processing
            if self.enable_memory_tracking:
                self._track_memory("save_start")
                
            # Use filename as base for cache files
            base_name = Path(file_path).stem
            text_path = self.text_dir / f"{base_name}.pt"
            image_path = self.image_dir / f"{base_name}.pt"
            latent_path = image_path  # Use image path for latents
            
            # Check and optimize device placement
            current_device = next(
                (t.device for t in latent_data.values() if isinstance(t, torch.Tensor)),
                torch.device('cpu')
            )
            
            # Create model dtypes configuration
            model_dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
            
            # Process tensors if present
            for tensor_dict in [latent_data, text_embeddings]:
                if tensor_dict is not None:
                    for k, v in tensor_dict.items():
                        if isinstance(v, torch.Tensor):
                            # Use appropriate dtype based on tensor type
                            target_dtype = (
                                model_dtypes.text_encoder.to_torch_dtype() 
                                if 'text' in k.lower() 
                                else model_dtypes.vae.to_torch_dtype()
                            )
                            if v.dtype != target_dtype:
                                tensor_dict[k] = v.to(dtype=target_dtype)

            # Memory optimization: Stream-based processing
            with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                # Pin memory for faster I/O if tensors present and support pinning
                for tensor_dict in [latent_data, text_embeddings]:
                    if tensor_dict is not None:
                        for tensor in tensor_dict.values():
                            if isinstance(tensor, torch.Tensor):
                                try:
                                    if not tensor.is_pinned() and tensor.device.type == 'cpu':
                                        pin_tensor_(tensor)
                                except Exception as e:
                                    logger.debug(f"Could not pin tensor memory: {str(e)}")
                                    # Continue without pinning if not supported
                                    continue
                            
                try:
                    # Save latents if present using chunked streaming
                    if latent_data is not None:
                        self._save_chunked_tensor(
                            {
                                "latent": latent_data,
                                "metadata": {
                                    **metadata,
                                    "latent_timestamp": time.time()
                                }
                            },
                            latent_path
                        )
                    
                    # Save text embeddings if present using chunked streaming
                    if text_embeddings is not None:
                        self._save_chunked_tensor(
                            {
                                "embeddings": text_embeddings,
                                "metadata": {
                                    **metadata,
                                    "text_timestamp": time.time()
                                }
                            },
                            text_path
                        )
                    
                    # Update index with filename-based paths
                    self.cache_index["files"][str(file_path)] = {
                        "latent_path": str(image_path),  # Use image path instead of latent
                        "text_path": str(text_path),
                        "base_name": base_name,
                        "timestamp": time.time()
                    }
                    
                    # Save index periodically
                    if len(self.cache_index["files"]) % 100 == 0:
                        self._save_cache_index()
                        
                finally:
                    # Cleanup: Unpin tensors and free memory if present and pinned
                    for tensor_dict in [latent_data, text_embeddings]:
                        if tensor_dict is not None:
                            for tensor in tensor_dict.values():
                                if isinstance(tensor, torch.Tensor) and tensor.is_pinned():
                                    try:
                                        unpin_tensor_(tensor)
                                    except Exception as e:
                                        logger.debug(f"Could not unpin tensor memory: {str(e)}")
                                        continue
                                
                    torch_sync()
                    
                # Track final memory state
                if self.enable_memory_tracking:
                    self._track_memory("save_complete")
                    
                return True
                
        except Exception as e:
            logger.error(f"Error saving preprocessed data for {file_path}: {str(e)}")
            if self.enable_memory_tracking:
                self._track_memory("save_error")
            return False

    def _process_image_batch(
        self,
        image_paths: List[Path],
        caption_ext: str,
        batch_size: int = 128  # Increased batch size
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Process batch of images with optimized parallel handling.
        
        Args:
            image_paths: List of paths to process
            caption_ext: Caption file extension
            batch_size: Size of processing batches
            
        Returns:
            Tuple of (tensors, metadata)
        """
        tensors = []
        metadata = {}
        
        # Split into batches for better memory efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_meta = {}
            
            # Pre-filter cached items
            to_process = [
                path for path in batch_paths 
                if str(path) not in self.cache_index["files"]
            ]
            
            if not to_process:
                self.stats.skipped_items += len(batch_paths)
                continue
            
            # Process batch in parallel with chunking
            chunks = [to_process[i:i + 16] for i in range(0, len(to_process), 16)]
            
            with ThreadPoolExecutor(max_workers=self.num_proc) as pool:
                futures = []
                for chunk in chunks:
                    futures.extend([
                        pool.submit(self._process_single_image, path, caption_ext)
                        for path in chunk
                    ])
                
                # Use torch.cuda.Stream for tensor operations
                stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                
                try:
                    with torch.cuda.stream(stream) if stream else nullcontext():
                        # Pre-allocate tensor storage
                        storage = []
                        meta_storage = []
                        
                        # Process results in chunks
                        for chunk_futures in [futures[i:i + 16] for i in range(0, len(futures), 16)]:
                            chunk_tensors = []
                            chunk_meta = {}
                            
                            for future in chunk_futures:
                                try:
                                    result = future.result()
                                    if result is not None:
                                        tensor, meta = result
                                        if self._validate_tensor(tensor):
                                            chunk_tensors.append(tensor)
                                            chunk_meta.update(meta)
                                            self.stats.processed_items += 1
                                        else:
                                            self.stats.validation_errors += 1
                                except Exception as e:
                                    self.stats.failed_items += 1
                                    logger.error(f"Batch processing error: {str(e)}")
                                    
                            # Stack chunk tensors
                            if chunk_tensors:
                                storage.append(torch.stack(chunk_tensors))
                                meta_storage.append(chunk_meta)
                                
                        # Combine all chunks
                        if storage:
                            batch_tensors = [torch.cat(storage)]
                            batch_meta = {k: v for d in meta_storage for k, v in d.items()}
                        
                        # Stack tensors if we have any
                        if batch_tensors:
                            stacked = torch.stack(batch_tensors)
                            if stream:
                                stacked.record_stream(stream)
                            tensors.append(stacked)
                            metadata.update(batch_meta)
                            
                finally:
                    if stream:
                        stream.synchronize()
                    
        return tensors, metadata

    def _validate_tensor(self, tensor: torch.Tensor) -> bool:
        """Validate tensor properties."""
        from ..utils.tensor_utils import validate_tensor
        return validate_tensor(
            tensor,
            expected_dims=4,
            enable_memory_tracking=self.enable_memory_tracking,
            memory_stats=self.memory_stats if hasattr(self, 'memory_stats') else None
        )

    def has_cached_item(self, file_path: Union[str, Path]) -> bool:
        """Check if item exists in cache.
        
        Args:
            file_path: Path to original file
            
        Returns:
            bool indicating if item is cached
        """
        try:
            # First check cache index
            file_info = self.cache_index["files"].get(str(file_path))
            if file_info:
                latent_path = Path(file_info["latent_path"])
                text_path = Path(file_info["text_path"])
                if latent_path.exists() and text_path.exists():
                    return True
                    
            # Check for files using base name pattern if not in index
            base_name = Path(file_path).stem
            potential_paths = [
                (self.image_dir / f"{base_name}.pt", self.text_dir / f"{base_name}.pt")
            ]
            
            for latent_path, text_path in potential_paths:
                if latent_path.exists() and text_path.exists():
                    # Update index with found files
                    self.cache_index["files"][str(file_path)] = {
                        "latent_path": str(latent_path),
                        "text_path": str(text_path),
                        "base_name": base_name,
                        "timestamp": time.time()
                    }
                    self._save_cache_index()
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache status: {str(e)}")
            return False

    def get_cached_item(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached item with optimized memory handling.
        
        Args:
            file_path: Path to original file
            device: Target device for tensors
            
        Returns:
            Dict containing cached tensors and metadata
        """
        try:
            if self.enable_memory_tracking:
                self._track_memory("cache_fetch_start")
                
            # First check cache index
            file_info = self.cache_index["files"].get(str(file_path))
            if not file_info:
                # Try finding files by image name pattern
                base_name = Path(file_path).stem
                potential_paths = [
                    (self.image_dir / f"{base_name}.pt", self.text_dir / f"{base_name}.pt")
                ]
                
                for latent_path, text_path in potential_paths:
                    if latent_path.exists() and text_path.exists():
                        file_info = {
                            "latent_path": str(latent_path),
                            "text_path": str(text_path),
                            "base_name": base_name,
                            "timestamp": time.time()
                        }
                        # Update index with found files
                        self.cache_index["files"][str(file_path)] = file_info
                        self._save_cache_index()
                        break
                        
                if not file_info:
                    self.stats.cache_misses += 1
                    return None
                    
            latent_path = Path(file_info["latent_path"])
            text_path = Path(file_info["text_path"])
            
            # Check if at least one type of data exists
            if not (latent_path.exists() or text_path.exists()):
                self.stats.corrupted_items += 1
                return None
                
            # Load data with memory optimization
            try:
                with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                    result = {}
                    metadata = {}
                    
                    # Load latents if they exist
                    if latent_path.exists():
                        latent_data = torch.load(latent_path, map_location='cpu')
                        result["latent"] = latent_data["latent"]
                        metadata.update(latent_data["metadata"])
                        
                        # Move to device if specified
                        if device is not None:
                            for k, v in result["latent"].items():
                                if isinstance(v, torch.Tensor):
                                    result["latent"][k] = v.to(device, non_blocking=True)
                    
                    # Load text embeddings if they exist
                    if text_path.exists():
                        text_data = torch.load(text_path, map_location='cpu')
                        result["text_embeddings"] = text_data["embeddings"]
                        metadata.update(text_data["metadata"])
                        
                        # Move to device if specified
                        if device is not None:
                            for k, v in result["text_embeddings"].items():
                                if isinstance(v, torch.Tensor):
                                    result["text_embeddings"][k] = v.to(device, non_blocking=True)
                    
                    self.stats.cache_hits += 1
                    result["metadata"] = metadata
                    return result
            finally:
                if self.enable_memory_tracking:
                    self._track_memory("cache_fetch_complete")
                    
        except Exception as e:
            logger.error(f"Error retrieving cached item: {str(e)}")
            self.stats.failed_items += 1
            return None

    def clear_cache(self, remove_files: bool = True):
        """Clear cache with proper cleanup."""
        try:
            if remove_files:
                # Remove cache directories
                import shutil
                for directory in [self.text_dir, self.image_dir]:
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir(parents=True)
                
            # Reset index
            self.cache_index = {"files": {}, "chunks": {}}
            self._save_cache_index()
            
            # Clear memory
            torch_sync()
            
            # Reset statistics
            self.stats = CacheStats()
            if self.enable_memory_tracking:
                self._init_memory_tracking()
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        try:
            self._save_cache_index()
        finally:
            self.image_pool.shutdown()
            self.io_pool.shutdown()
            torch_sync()
            
    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """Get aspect ratio buckets from config."""
        return config.global_config.image.supported_dims
        
    def assign_aspect_buckets(
        self,
        image_paths: List[str],
        buckets: List[Tuple[int, int]],
        max_aspect_ratio: float,
        batch_size: int = 256  # Increased batch size
    ) -> List[int]:
        """Assign images to aspect ratio buckets with batched processing."""
        # Initialize bucket assignment cache
        if not hasattr(self, '_bucket_cache'):
            self._bucket_cache = {}
            
        bucket_indices = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_indices = []
            
            # Filter uncached paths
            to_process = [
                path for path in batch_paths
                if path not in self._bucket_cache
            ]
            
            if to_process:
                # Process uncached images in parallel with chunking
                chunks = [to_process[i:i + 32] for i in range(0, len(to_process), 32)]
            
                with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as pool:
                    all_futures = []
                    for chunk in chunks:
                        chunk_futures = [
                            pool.submit(
                                self._assign_single_bucket,
                                img_path,
                                buckets,
                                max_aspect_ratio
                            )
                            for img_path in chunk
                        ]
                        all_futures.extend(chunk_futures)
                    futures = all_futures
                    
                    # Update cache with new results
                    for path, future in zip(to_process, futures):
                        try:
                            bucket_idx = future.result()
                            self._bucket_cache[path] = bucket_idx
                        except Exception as e:
                            logger.error(f"Error in bucket assignment: {str(e)}")
                            self._bucket_cache[path] = 0
            
            # Collect results from cache
            batch_indices = [
                self._bucket_cache.get(path, 0)
                for path in batch_paths
            ]
            
            bucket_indices.extend(batch_indices)
            
        return bucket_indices
        
    def _assign_single_bucket(
        self,
        img_path: str,
        buckets: List[Tuple[int, int]],
        max_aspect_ratio: float
    ) -> int:
        """Assign single image to best matching bucket."""
        try:
            img = Image.open(img_path)
            w, h = img.size
            aspect_ratio = w / h
            img_area = w * h
            
            # Find best bucket match
            min_diff = float('inf')
            best_idx = 0
            
            for idx, (bucket_h, bucket_w) in enumerate(buckets):
                bucket_ratio = bucket_w / bucket_h
                if bucket_ratio > max_aspect_ratio:
                    continue
                    
                ratio_diff = abs(aspect_ratio - bucket_ratio)
                area_diff = abs(img_area - (bucket_w * bucket_h))
                total_diff = (ratio_diff * 2.0) + (area_diff / (1536 * 1536))
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_idx = idx
                    
            return best_idx
            
        except Exception as e:
            logger.error(f"Error assigning bucket for {img_path}: {str(e)}")
            return 0
    def _save_chunked_tensor(self, data: Dict, path: Path) -> None:
        """Save tensor data in chunks with streaming.
        
        Args:
            data: Dictionary containing tensors and metadata
            path: Output file path
        """
        # Calculate optimal chunk size based on available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            chunk_bytes = int(total_memory * self.max_chunk_memory)
        else:
            chunk_bytes = self.stream_buffer_size
            
        # Stream data in chunks
        with open(path, 'wb') as f:
            for tensor_dict in self._chunk_tensor_dict(data, chunk_bytes):
                # Save directly to file, using memory mapping for large tensors
                torch.save(
                    tensor_dict,
                    f,
                    _use_new_zipfile_serialization=True
                )
                    
    def _chunk_tensor_dict(self, data: Dict, chunk_bytes: int):
        """Generate chunks of tensor dictionary.
        
        Args:
            data: Dictionary containing tensors
            chunk_bytes: Maximum bytes per chunk
            
        Yields:
            Dictionary chunks
        """
        # Split tensors into chunks
        chunks = []
        current_bytes = 0
        current_chunk = {}
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensor_bytes = value.nelement() * value.element_size()
                if tensor_bytes > chunk_bytes:
                    # Split large tensor
                    splits = torch.split(value, chunk_bytes // value.element_size())
                    for i, split in enumerate(splits):
                        chunk = {
                            f"{key}_chunk_{i}": split,
                            "metadata": data.get("metadata", {})
                        }
                        yield chunk
                else:
                    if current_bytes + tensor_bytes > chunk_bytes:
                        yield current_chunk
                        current_chunk = {}
                        current_bytes = 0
                    current_chunk[key] = value
                    current_bytes += tensor_bytes
            else:
                current_chunk[key] = value
                
        if current_chunk:
            yield current_chunk
