"""High-performance cache management with optimized tensor and memory handling."""
import multiprocessing as mp
import traceback
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_,
    torch_gc
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
        
        # Setup cache paths
        self.latents_dir = self.cache_dir / "latents"
        self.text_dir = self.cache_dir / "text"
        self.latents_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        
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
            
        # Setup worker pools with proper resource limits
        self.image_pool = ProcessPoolExecutor(
            max_workers=self.num_proc,
            mp_context=mp.get_context('spawn')
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.num_proc * 2,
            thread_name_prefix="cache_io"
        )
        
        # Initialize memory tracking if enabled
        if self.enable_memory_tracking and torch.cuda.is_available():
            self._init_memory_tracking()

    def _init_memory_tracking(self):
        """Initialize memory tracking utilities."""
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
        latent_data: Dict[str, torch.Tensor],
        text_embeddings: Dict[str, torch.Tensor],
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
            
            # Memory optimization: Stream-based processing
            with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                # Pin memory for faster I/O
                for tensor_dict in [latent_data, text_embeddings]:
                    for tensor in tensor_dict.values():
                        if isinstance(tensor, torch.Tensor):
                            pin_tensor_(tensor)
                            
                try:
                    # Save latents and text embeddings separately for better memory management
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
                                
                    torch_gc()
                    
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
        """Validate tensor properties with memory tracking."""
        try:
            if self.enable_memory_tracking:
                self._track_memory("validate_start")
                
            if not isinstance(tensor, torch.Tensor):
                return False
                
            if tensor.dim() != 4:  # [B, C, H, W]
                return False
                
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
                
            # Check tensor device placement
            if tensor.device.type == 'cuda':
                # Ensure tensor is in contiguous memory
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                    
            return True
            
        finally:
            if self.enable_memory_tracking:
                self._track_memory("validate_complete")

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
                
            latent_path = Path(file_info["latent_path"])
            text_path = Path(file_info["text_path"])
            
            if not latent_path.exists() or not text_path.exists():
                self.stats.corrupted_items += 1
                return None
                
            # Load data with memory optimization
            try:
                with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                    # Load latents
                    latent_data = torch.load(latent_path, map_location='cpu')
                    text_data = torch.load(text_path, map_location='cpu')
                    
                    # Move to target device if specified
                    if device is not None:
                        for tensor_dict in [latent_data["latent"], text_data["embeddings"]]:
                            for k, v in tensor_dict.items():
                                if isinstance(v, torch.Tensor):
                                    tensor_dict[k] = v.to(device, non_blocking=True)
                                    
                    self.stats.cache_hits += 1
                    
                    return {
                        "latent": latent_data["latent"],
                        "text_embeddings": text_data["embeddings"],
                        "metadata": {
                            **latent_data["metadata"],
                            **text_data["metadata"]
                        }
                    }
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
            torch_gc()
            
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
            torch_gc()
