"""High-performance cache management for large-scale dataset preprocessing."""
import multiprocessing as mp
import traceback
import time
from dataclasses import dataclass
from enum import Enum, auto
from src.core.logging.logging import setup_logging
from ..utils.paths import convert_windows_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_
)
import hashlib
import json
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

class CacheError(Exception):
    """Base exception for cache-related errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class CacheIOError(CacheError):
    """Raised when cache I/O operations fail."""
    pass

class CacheValidationError(CacheError):
    """Raised when cache validation fails."""
    pass

class CacheProcessingError(CacheError):
    """Raised when processing cache items fails."""
    pass

class CacheStage(Enum):
    """Enum for tracking cache processing stages."""
    INITIALIZATION = auto()
    IMAGE_PROCESSING = auto()
    TENSOR_VALIDATION = auto()
    CHUNK_SAVING = auto()
    INDEX_UPDATE = auto()

@dataclass
class CacheStats:
    """Statistics for cache operations."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    corrupted_items: int = 0
    validation_errors: int = 0

logger = setup_logging(__name__, level="INFO")

class CacheManager:
    """Manages high-throughput caching of image-caption pairs."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        num_proc: Optional[int] = None,
        chunk_size: int = 1000,
        compression: Optional[str] = "zstd",
        verify_hashes: bool = True
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cached files
            num_proc: Number of processes (default: CPU count)
            chunk_size: Number of items per cache chunk
            compression: Compression algorithm (None, 'zstd', 'gzip')
            verify_hashes: Whether to verify content hashes
        """
        self.cache_dir = Path(convert_windows_path(cache_dir, make_absolute=True))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_proc = num_proc or mp.cpu_count()
        self.chunk_size = chunk_size
        self.compression = compression
        self.verify_hashes = verify_hashes
        
        # Setup cache index
        self.index_path = Path(convert_windows_path(self.cache_dir / "cache_index.json", make_absolute=True))
        self.cache_index = self._load_cache_index()
        
        # Setup process pools
        self.image_pool = ProcessPoolExecutor(max_workers=self.num_proc)
        self.io_pool = ThreadPoolExecutor(max_workers=self.num_proc * 2)
        
    def _load_cache_index(self) -> Dict:
        """Load or create cache index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"files": {}, "chunks": {}}
        
    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f)
            
    def _compute_hash(self, file_path: Path) -> str:
        """Compute file hash for verification."""
        hasher = hashlib.sha256()
        converted_path = convert_windows_path(file_path, make_absolute=True)
        with open(converted_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def _process_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Process single image with optimized memory handling and CUDA streams.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed tensor or None if processing failed
            
        Raises:
            CacheProcessingError: If image processing fails
        """
        try:
            # Create streams for pipelined operations
            compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            
            # Use NVIDIA DALI for faster image loading if available
            try:
                import nvidia.dali as dali
                import nvidia.dali.fn as fn
                use_dali = True
            except ImportError:
                use_dali = False
                
            if use_dali:
                pipe = dali.Pipeline(batch_size=1, num_threads=1, device_id=0)
                with pipe:
                    images = fn.readers.file(name="Reader", files=[str(image_path)])
                    decoded = fn.decoders.image(images, device="mixed")
                    normalized = fn.crop_mirror_normalize(
                        decoded,
                        dtype=dali.types.FLOAT,
                        mean=[0.5 * 255] * 3,
                        std=[0.5 * 255] * 3,
                        output_layout="CHW"
                    )
                    pipe.set_outputs(normalized)
                pipe.build()
                
                # Use compute stream for DALI processing
                if compute_stream is not None:
                    with torch.cuda.stream(compute_stream):
                        tensor = pipe.run()[0].as_tensor()
                else:
                    tensor = pipe.run()[0].as_tensor()
            else:
                # Fallback to PIL with optimizations
                image = Image.open(image_path).convert('RGB')
                tensor = torch.from_numpy(np.array(image)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # CHW format
                
            # Optimize memory format and transfer
            if transfer_stream is not None:
                with torch.cuda.stream(transfer_stream):
                    tensor = tensor.contiguous(memory_format=torch.channels_last)
                    if torch.cuda.is_available():
                        tensor = tensor.pin_memory()
                        tensors_record_stream(transfer_stream, tensor)
            else:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                    
            # Synchronize streams if used
            if compute_stream is not None:
                compute_stream.synchronize()
            if transfer_stream is not None:
                transfer_stream.synchronize()
                
            return tensor
        except Exception as e:
            error_context = {
                'image_path': str(image_path),
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'traceback': traceback.format_exc(),
                'stage': CacheStage.IMAGE_PROCESSING.name
            }
            logger.error(
                f"Error processing image:\n"
                f"Image path: {error_context['image_path']}\n"
                f"Error type: {error_context['error_type']}\n"
                f"Error message: {error_context['error_msg']}\n"
                f"Processing stage: {error_context['stage']}\n"
                f"Traceback:\n{error_context['traceback']}"
            )
            raise CacheProcessingError("Failed to process image", context=error_context)
            
    def _save_chunk(
        self,
        chunk_id: int,
        tensors: List[torch.Tensor],
        metadata: Dict,
        cache_dir: Path
    ) -> bool:
        """Save chunk of processed tensors to disk.
        
        Args:
            chunk_id: Chunk identifier
            tensors: List of tensors to save
            metadata: Associated metadata
            cache_dir: Cache directory path
            
        Returns:
            bool indicating success
        """
        try:
            # Create chunk directory
            chunk_dir = cache_dir / f"chunk_{chunk_id}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tensors with compression
            chunk_path = chunk_dir / "data.pt"
            torch.save(
                {"tensors": tensors, "metadata": metadata},
                chunk_path,
                _use_new_zipfile_serialization=True
            )
            
            # Update index
            self.cache_index["chunks"][str(chunk_id)] = {
                "path": str(chunk_path),
                "size": chunk_path.stat().st_size,
                "timestamp": time.time()
            }
            
            return True
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_id}: {str(e)}")
            return False
            
    def process_dataset(
        self,
        data_dir: Union[str, Path],
        image_exts: List[str] = [".jpg", ".jpeg", ".png"],
        caption_ext: str = ".txt", 
        num_workers: Optional[int] = None
    ) -> CacheStats:
        """Process dataset with performance metrics logging.
        
        Args:
            data_dir: Directory containing image-caption pairs
            image_exts: List of valid image extensions
            caption_ext: Caption file extension
            num_workers: Number of worker processes
            
        Returns:
            Dict with processing statistics
        """
        logger.info(f"Starting dataset processing with {num_workers or self.num_proc} workers")
        from ..utils.paths import convert_windows_path, is_wsl
        data_dir = convert_windows_path(data_dir, make_absolute=True)
        if is_wsl():
            logger.info(f"Running in WSL, using converted path: {data_dir}")
        """Process entire dataset with parallel processing.
        
        Args:
            data_dir: Directory containing image-caption pairs
            image_exts: List of valid image extensions
            caption_ext: Caption file extension
            
        Returns:
            Processing statistics
        """
        data_dir = Path(data_dir)
        stats = CacheStats()
        
        # Get all image files with WSL path handling
        image_files = []
        for ext in image_exts:
            found_files = list(data_dir.glob(f"*{ext}"))
            # Convert any Windows paths and handle list inputs
            for f in found_files:
                if isinstance(f, (list, tuple)):
                    # Take first path if given a list
                    if f:
                        path = f[0] if isinstance(f[0], (str, Path)) else str(f[0])
                        image_files.append(Path(str(convert_windows_path(path, make_absolute=True))))
                else:
                    image_files.append(Path(str(convert_windows_path(f, make_absolute=True))))
            
        logger.info(f"Found {len(image_files)} images to process")
        
        # Use provided num_workers or default
        workers = num_workers if num_workers is not None else self.num_proc
        
        # Process in chunks
        for chunk_start in tqdm(range(0, len(image_files), self.chunk_size), 
                               desc="Processing chunks",
                               disable=workers > 1):
            chunk = image_files[chunk_start:chunk_start + self.chunk_size]
            chunk_id = chunk_start // self.chunk_size
            
            # Process images in parallel
            futures = []
            for img_path in chunk:
                if str(img_path) in self.cache_index["files"]:
                    stats.skipped_items += 1
                    logger.debug(f"Skipping already cached image: {img_path}")
                    continue
                
                caption_path = img_path.with_suffix(caption_ext)
                if not caption_path.exists():
                    error_context = {
                        'image_path': str(img_path),
                        'caption_path': str(caption_path),
                        'stage': CacheStage.INITIALIZATION.name
                    }
                    logger.error(
                        f"Missing caption file:\n"
                        f"Image path: {error_context['image_path']}\n"
                        f"Caption path: {error_context['caption_path']}\n"
                        f"Stage: {error_context['stage']}"
                    )
                    stats.failed_items += 1
                    continue
                
                try:
                    futures.append(
                        self.image_pool.submit(
                            self._process_image, 
                            img_path
                        ) if workers > 1 else self._process_image(img_path)
                    )
                except Exception as e:
                    error_context = {
                        'image_path': str(img_path),
                        'error_type': type(e).__name__,
                        'error_msg': str(e),
                        'traceback': traceback.format_exc(),
                        'stage': CacheStage.INITIALIZATION.name
                    }
                    logger.error(
                        f"Error submitting processing job:\n"
                        f"Image path: {error_context['image_path']}\n"
                        f"Error type: {error_context['error_type']}\n"
                        f"Error message: {error_context['error_msg']}\n"
                        f"Stage: {error_context['stage']}\n"
                        f"Traceback:\n{error_context['traceback']}"
                    )
                    stats.failed_items += 1
                
            # Collect results
            tensors = []
            metadata = {}
            
            for i, future in enumerate(futures):
                try:
                    tensor = future.result()
                    try:
                        img_path = chunk[i]
                        if isinstance(img_path, (list, tuple)):
                            # Handle list of paths by taking first valid one
                            valid_path = None
                            for p in img_path:
                                try:
                                    test_path = Path(str(p))
                                    if test_path.exists():
                                        valid_path = test_path
                                        break
                                except Exception:
                                    continue
                            if valid_path is None:
                                raise ValueError(f"No valid path found in list: {img_path}")
                            img_path = valid_path
                        else:
                            img_path = Path(str(img_path))
                
                        caption_path = img_path.with_suffix(caption_ext)
                
                        # Read caption with error handling
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    except Exception as e:
                        error_context = {
                            'image_path': str(img_path),
                            'caption_path': str(caption_path),
                            'error_type': type(e).__name__,
                            'error_msg': str(e),
                            'traceback': traceback.format_exc(),
                            'stage': CacheStage.INITIALIZATION.name
                        }
                        logger.error(
                            f"Error reading caption file:\n"
                            f"Image path: {error_context['image_path']}\n"
                            f"Caption path: {error_context['caption_path']}\n"
                            f"Error type: {error_context['error_type']}\n"
                            f"Error message: {error_context['error_msg']}\n"
                            f"Stage: {error_context['stage']}\n"
                            f"Traceback:\n{error_context['traceback']}"
                        )
                        stats.failed_items += 1
                        continue
                        
                    # Validate tensor
                    try:
                        if not isinstance(tensor, torch.Tensor):
                            raise CacheValidationError(
                                f"Invalid tensor type: {type(tensor)}",
                                {'image_path': str(img_path)}
                            )
                        if tensor.dim() != 4:  # [C, H, W] or [B, C, H, W]
                            raise CacheValidationError(
                                f"Invalid tensor dimensions: {tensor.dim()}",
                                {'image_path': str(img_path)}
                            )
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            raise CacheValidationError(
                                "Tensor contains NaN or Inf values",
                                {'image_path': str(img_path)}
                            )
                    except CacheValidationError as e:
                        logger.error(
                            f"Tensor validation failed:\n"
                            f"Image path: {e.context.get('image_path')}\n"
                            f"Error: {str(e)}"
                        )
                        stats.validation_errors += 1
                        continue
                        
                    # Store tensor and metadata
                    tensors.append(tensor)
                    metadata[str(img_path)] = {
                        "caption": caption,
                        "hash": self._compute_hash(img_path) if self.verify_hashes else None
                    }
                    stats.processed_items += 1
                    logger.debug(f"Successfully processed: {img_path}")
                except Exception as e:
                    error_context = {
                        'image_path': str(chunk[i]),
                        'error_type': type(e).__name__,
                        'error_msg': str(e),
                        'traceback': traceback.format_exc(),
                        'stage': CacheStage.IMAGE_PROCESSING.name
                    }
                    logger.error(
                        f"Error processing chunk item:\n"
                        f"Image path: {error_context['image_path']}\n"
                        f"Error type: {error_context['error_type']}\n"
                        f"Error message: {error_context['error_msg']}\n"
                        f"Stage: {error_context['stage']}\n"
                        f"Traceback:\n{error_context['traceback']}"
                    )
                    stats.failed_items += 1
                    
            # Save chunk if not empty
            if tensors:
                if self._save_chunk(chunk_id, tensors, metadata):
                    # Update main index
                    for img_path, meta in metadata.items():
                        self.cache_index["files"][img_path] = {
                            "chunk_id": chunk_id,
                            "metadata": meta
                        }
                        
            # Save index periodically
            if chunk_id % 10 == 0:
                self._save_cache_index()
                
        # Final index save
        self._save_cache_index()
        
        return stats
        
    def get_cached_item(
        self,
        image_path: Union[str, Path, List[Union[str, Path]]]
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """Retrieve cached item by image path."""
        # Convert input to Path object
        try:
            if isinstance(image_path, (list, tuple)):
                # Try each path in the list until we find a valid one
                for p in image_path:
                    try:
                        converted_path = convert_windows_path(p, make_absolute=True)
                        if Path(converted_path).exists():
                            image_path = converted_path
                            break
                    except Exception:
                        continue
                else:  # No valid path found
                    return None
            else:
                image_path = convert_windows_path(image_path, make_absolute=True)
        except Exception as e:
            logger.warning(f"Error converting path: {str(e)}")
            return None
        
        if image_path not in self.cache_index["files"]:
            return None
            
        file_info = self.cache_index["files"][image_path]
        chunk_id = file_info["chunk_id"]
        chunk_info = self.cache_index["chunks"][str(chunk_id)]
        
        # Load chunk
        chunk_path = Path(chunk_info["path"])
        if not chunk_path.exists():
            return None
            
        tensors = torch.load(chunk_path)
        caption = file_info["metadata"]["caption"]
        
        return tensors, caption
        
    def verify_cache(self) -> Dict[str, int]:
        """Verify cache integrity."""
        stats = {"valid": 0, "corrupted": 0, "missing": 0}
        
        for file_path, file_info in tqdm(self.cache_index["files"].items()):
            if not self.verify_hashes:
                continue
                
            try:
                current_hash = self._compute_hash(file_path)
                stored_hash = file_info["metadata"]["hash"]
                
                if current_hash == stored_hash:
                    stats["valid"] += 1
                else:
                    stats["corrupted"] += 1
            except FileNotFoundError:
                stats["missing"] += 1
                
        return stats
        
    def clear_cache(self):
        """Clear all cached files."""
        # Remove chunk files
        for chunk_id, chunk_info in self.cache_index["chunks"].items():
            try:
                Path(chunk_info["path"]).unlink()
            except FileNotFoundError:
                logger.warning(f"Cache chunk {chunk_id} already missing")
                
        # Clear index
        self.cache_index = {"files": {}, "chunks": {}}
        self._save_cache_index()
