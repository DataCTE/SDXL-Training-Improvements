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
        enable_memory_tracking: bool = True
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
        self.latents_dir = self.cache_dir / "latents"
        self.text_dir = self.cache_dir / "text"
        self.latents_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        
        # Setup worker pools with proper resource limits
        self.image_pool = ProcessPoolExecutor(
            max_workers=self.num_proc,
            mp_context=mp.get_context('spawn')
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.num_proc * 2,
            thread_name_prefix="cache_io"
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
                
            # Create cache paths
            file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:12]
            latent_path = self.latents_dir / f"{file_hash}_latent.pt"
            text_path = self.text_dir / f"{file_hash}_text.pt"
            
            # Check and optimize device placement
            current_device = next(
                (t.device for t in latent_data.values() if isinstance(t, torch.Tensor)),
                torch.device('cpu')
            )
            
            # Create model dtypes configuration
            model_dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
            
            for tensor_dict in [latent_data, text_embeddings]:
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
                # Pin memory for faster I/O
                for tensor_dict in [latent_data, text_embeddings]:
                    for tensor in tensor_dict.values():
                        if isinstance(tensor, torch.Tensor):
                            pin_tensor_(tensor)
                            
                try:
                    # Save latents if present
                    if latent_data is not None:
                        torch.save(
                            {
                                "latent": latent_data,
                                "metadata": {
                                    **metadata,
                                    "latent_timestamp": time.time()
                                }
                            },
                            latent_path,
                            _use_new_zipfile_serialization=True
                        )
                    
                    # Save text embeddings if present
                    if text_embeddings is not None:
                        torch.save(
                            {
                                "embeddings": text_embeddings,
                                "metadata": {
                                    **metadata,
                                    "text_timestamp": time.time()
                                }
                            },
                            text_path,
                            _use_new_zipfile_serialization=True
                        )
                    
                    # Update index
                    self.cache_index["files"][str(file_path)] = {
                        "latent_path": str(latent_path),
                        "text_path": str(text_path),
                        "hash": file_hash,
                        "timestamp": time.time()
                    }
                    
                    # Save index periodically
                    if len(self.cache_index["files"]) % 100 == 0:
                        self._save_cache_index()
                        
                finally:
                    # Cleanup: Unpin tensors and free memory
                    for tensor_dict in [latent_data, text_embeddings]:
                        for tensor in tensor_dict.values():
                            if isinstance(tensor, torch.Tensor):
                                unpin_tensor_(tensor)
                                
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
        caption_ext: str
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Process batch of images with optimized memory handling.
        
        Args:
            image_paths: List of paths to process
            caption_ext: Caption file extension
            
        Returns:
            Tuple of (tensors, metadata)
        """
        tensors = []
        metadata = {}
        
        # Use thread pool for parallel I/O
        with ThreadPoolExecutor(max_workers=self.num_proc) as pool:
            futures = []
            
            # Submit processing jobs
            for path in image_paths:
                if str(path) in self.cache_index["files"]:
                    self.stats.skipped_items += 1
                    continue
                    
                futures.append(pool.submit(self._process_single_image, path, caption_ext))
                
            # Collect results with proper error handling
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        tensor, meta = result
                        # Validate tensor
                        if self._validate_tensor(tensor):
                            tensors.append(tensor)
                            metadata.update(meta)
                            self.stats.processed_items += 1
                        else:
                            self.stats.validation_errors += 1
                except Exception as e:
                    self.stats.failed_items += 1
                    logger.error(f"Batch processing error: {str(e)}")
                    
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
            file_info = self.cache_index["files"].get(str(file_path))
            if not file_info:
                return False
                
            latent_path = Path(file_info["latent_path"])
            text_path = Path(file_info["text_path"])
            
            return latent_path.exists() and text_path.exists()
            
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
                
            file_info = self.cache_index["files"].get(str(file_path))
            if not file_info:
                self.stats.cache_misses += 1
                return None
                
            latent_path = Path(file_info.get("latent_path", ""))
            text_path = Path(file_info.get("text_path", ""))
            
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
                shutil.rmtree(self.latents_dir)
                shutil.rmtree(self.text_dir)
                
                # Recreate directories
                self.latents_dir.mkdir(parents=True)
                self.text_dir.mkdir(parents=True)
                
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
        max_aspect_ratio: float
    ) -> List[int]:
        """Assign images to aspect ratio buckets."""
        bucket_indices = []
        
        # Process images in parallel using worker pool
        with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as pool:
            futures = []
            
            for img_path in image_paths:
                futures.append(pool.submit(
                    self._assign_single_bucket,
                    img_path,
                    buckets,
                    max_aspect_ratio
                ))
                
            for future in futures:
                try:
                    bucket_idx = future.result()
                    bucket_indices.append(bucket_idx)
                except Exception as e:
                    logger.error(f"Error in bucket assignment: {str(e)}")
                    bucket_indices.append(0)
                    
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
