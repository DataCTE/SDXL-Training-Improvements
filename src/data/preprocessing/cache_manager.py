"""High-performance cache management with extreme speedups."""
import multiprocessing as mp
import logging
import multiprocessing as mp
import traceback
import time
import hashlib
import json
import torch
import torch.backends.cudnn
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image


from tqdm.auto import tqdm
from contextlib import nullcontext

from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_,
    torch_sync
)
from src.core.logging.logging import setup_logging
from ..utils.paths import convert_windows_path

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
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
    """Manages high-throughput caching with extreme memory and speed optimizations."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        num_proc: Optional[int] = None,
        chunk_size: int = 1000,
        compression: Optional[str] = "zstd",
        verify_hashes: bool = True,
        max_memory_usage: float = 0.8,
        enable_memory_tracking: bool = True,
        stream_buffer_size: int = 1024 * 1024,
        max_chunk_memory: float = 0.2
    ):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        self.cache_dir = Path(convert_windows_path(cache_dir, make_absolute=True))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_proc = num_proc or mp.cpu_count()
        self.chunk_size = chunk_size
        self.compression = compression
        self.verify_hashes = verify_hashes
        self.max_memory_usage = max_memory_usage
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_buffer_size = stream_buffer_size
        self.max_chunk_memory = max_chunk_memory
        self.stats = CacheStats()
        self.memory_stats = {
            'peak_allocated': 0,
            'total_allocated': 0,
            'num_allocations': 0,
            'oom_events': 0
        }
        self.text_dir = self.cache_dir / "text"
        self.image_dir = self.cache_dir / "image"
        for directory in [self.text_dir, self.image_dir]:
            directory.mkdir(exist_ok=True)
        self.image_pool = ProcessPoolExecutor(
            max_workers=self.num_proc,
            mp_context=mp.get_context('spawn')  # Ensures 'spawn' start method
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.num_proc * 2,
            thread_name_prefix="cache_io"
        )
        if torch.cuda.is_available():
            self.pinned_buffer = torch.empty((stream_buffer_size,), dtype=torch.uint8, pin_memory=True)
        self.index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        # Disable torch.compile for now due to logging issues

    def _load_cache_index(self) -> Dict:
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            return {"files": {}, "chunks": {}}
        except Exception as e:
            logger.error(f"Failed to load cache index: {str(e)}")
            return {"files": {}, "chunks": {}}
            
    def _save_cache_index(self) -> None:
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {str(e)}")

    def _init_memory_tracking(self):
        if not hasattr(self, 'memory_stats'):
            self.memory_stats = {
                'peak_allocated': 0,
                'total_allocated': 0,
                'num_allocations': 0,
                'oom_events': 0
            }

    def _track_memory(self, context: str):
        if self.enable_memory_tracking and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                self.memory_stats['peak_allocated'] = max(self.memory_stats['peak_allocated'], allocated)
                self.memory_stats['total_allocated'] += allocated
                self.memory_stats['num_allocations'] += 1
                if allocated > self.max_memory_usage * torch.cuda.get_device_properties(0).total_memory:
                    logger.warning(f"High memory usage in {context}: {allocated / 1e9:.2f}GB")
            except Exception as e:
                logger.error(f"Memory tracking error in {context}: {str(e)}")

    def load_preprocessed_data(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load preprocessed data from cache.
        
        Args:
            file_path: Path to original image file
            
        Returns:
            Dictionary containing cached data if available, None otherwise
        """
        return self.get_cached_item(file_path)
        
    def save_preprocessed_data(
        self,
        latent_data: Optional[Dict[str, torch.Tensor]],
        text_embeddings: Optional[Dict[str, torch.Tensor]],
        metadata: Dict,
        file_path: Union[str, Path]
    ) -> bool:
        try:
            if latent_data is None and text_embeddings is None:
                logger.error(f"Both latent_data and text_embeddings are None for {file_path}")
                return False
            if self.enable_memory_tracking:
                self._track_memory("save_start")
            base_name = Path(file_path).stem
            text_path = self.text_dir / f"{base_name}.pt"
            image_path = self.image_dir / f"{base_name}.pt"
            latent_path = image_path
            current_device = next(
                (t.device for t in latent_data.values() if isinstance(t, torch.Tensor)),
                torch.device('cpu')
            )
            model_dtypes = ModelWeightDtypes.from_single_dtype(DataType.FLOAT_32)
            try:
                if latent_data is not None:
                    try:
                        processed_latents = {}
                        for k, v in latent_data.items():
                            if isinstance(v, torch.Tensor):
                                tensor = v.detach()
                                if tensor.device.type != 'cpu':
                                    tensor = tensor.cpu()
                                target_dtype = model_dtypes.vae.to_torch_dtype()
                                if tensor.dtype != target_dtype:
                                    tensor = tensor.to(dtype=target_dtype)
                                processed_latents[k] = tensor
                            else:
                                processed_latents[k] = v
                        save_data = {
                            "latent": processed_latents,
                            "metadata": {**metadata, "latent_timestamp": time.time()}
                        }
                        latent_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(save_data, latent_path)
                        if not latent_path.exists():
                            raise IOError(f"Failed to write latent file: {latent_path}")
                    except Exception as e:
                        logger.error(f"Error saving latent data: {str(e)}")
                        raise
                if text_embeddings is not None:
                    processed_embeddings = {}
                    for k, v in text_embeddings.items():
                        if isinstance(v, torch.Tensor):
                            target_dtype = model_dtypes.text_encoder.to_torch_dtype()
                            processed_embeddings[k] = v.detach().cpu().to(dtype=target_dtype)
                        else:
                            processed_embeddings[k] = v
                    torch.save(
                        {
                            "embeddings": processed_embeddings,
                            "metadata": {**metadata, "text_timestamp": time.time()}
                        },
                        text_path
                    )
                self.cache_index["files"][str(file_path)] = {
                    "latent_path": str(latent_path),
                    "text_path": str(text_path),
                    "base_name": base_name,
                    "timestamp": time.time()
                }
                self._save_cache_index()
                return True
            finally:
                for tensor_dict in [latent_data, text_embeddings]:
                    if tensor_dict is not None:
                        for tensor in tensor_dict.values():
                            if isinstance(tensor, torch.Tensor) and tensor.is_pinned():
                                try:
                                    unpin_tensor_(tensor)
                                except Exception as e:
                                    logger.debug(f"Could not unpin tensor: {str(e)}")
                                    continue
                torch_sync()
                if self.enable_memory_tracking:
                    self._track_memory("save_complete")
        except Exception as e:
            logger.error(f"Error saving preprocessed data for {file_path}: {str(e)}")
            if self.enable_memory_tracking:
                self._track_memory("save_error")
            return False

    def _process_image_batch(
        self,
        image_paths: List[Path],
        caption_ext: str,
        batch_size: int = 128
    ) -> Tuple[List[torch.Tensor], Dict]:
        tensors = []
        metadata = {}
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_meta = {}
            to_process = [
                path for path in batch_paths 
                if str(path) not in self.cache_index["files"]
            ]
            if not to_process:
                self.stats.skipped_items += len(batch_paths)
                continue
            chunks = [to_process[i:i + 16] for i in range(0, len(to_process), 16)]
            with ThreadPoolExecutor(max_workers=self.num_proc) as pool:
                futures = []
                for chunk in chunks:
                    futures.extend([
                        pool.submit(self._process_single_image, path, caption_ext)
                        for path in chunk
                    ])
                stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                try:
                    with torch.cuda.stream(stream) if stream else nullcontext():
                        storage = []
                        meta_storage = []
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
                            if chunk_tensors:
                                storage.append(torch.stack(chunk_tensors))
                                meta_storage.append(chunk_meta)
                        if storage:
                            batch_tensors = [torch.cat(storage)]
                            batch_meta = {k: v for d in meta_storage for k, v in d.items()}
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
        from ..utils.tensor_utils import validate_tensor
        return validate_tensor(
            tensor,
            expected_dims=4,
            enable_memory_tracking=self.enable_memory_tracking,
            memory_stats=self.memory_stats if hasattr(self, 'memory_stats') else None
        )

    def has_cached_item(self, file_path: Union[str, Path]) -> bool:
        try:
            str_path = str(file_path)
            return str_path in self.cache_index["files"]
        except Exception as e:
            logger.error(f"Error checking cache status: {str(e)}")
            return False

    def get_cached_item(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            if self.enable_memory_tracking:
                self._track_memory("cache_fetch_start")
            file_info = self.cache_index["files"].get(str(file_path))
            if not file_info:
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
                        self.cache_index["files"][str(file_path)] = file_info
                        self._save_cache_index()
                        break
                if not file_info:
                    self.stats.cache_misses += 1
                    return None
            latent_path = Path(file_info["latent_path"])
            text_path = Path(file_info["text_path"])
            if not (latent_path.exists() or text_path.exists()):
                self.stats.corrupted_items += 1
                return None
            try:
                with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                    result = {}
                    metadata = {}
                    if latent_path.exists():
                        latent_data = torch.load(latent_path, map_location='cpu')
                        result["latent"] = latent_data["latent"]
                        metadata.update(latent_data["metadata"])
                        if device is not None:
                            for k, v in result["latent"].items():
                                if isinstance(v, torch.Tensor):
                                    result["latent"][k] = v.to(device, non_blocking=True)
                    if text_path.exists():
                        text_data = torch.load(text_path, map_location='cpu')
                        result["text_embeddings"] = text_data["embeddings"]
                        metadata.update(text_data["metadata"])
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
            logger.error(f"Error retrieving cached item: {str(e)}", exc_info=True)
            self.stats.failed_items += 1
            return None

    def clear_cache(self, remove_files: bool = True):
        try:
            if remove_files:
                import shutil
                for directory in [self.text_dir, self.image_dir]:
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir(parents=True)
            self.cache_index = {"files": {}, "chunks": {}}
            self._save_cache_index()
            torch_sync()
            self.stats = CacheStats()
            if self.enable_memory_tracking:
                self._init_memory_tracking()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._save_cache_index()
        finally:
            self.image_pool.shutdown()
            self.io_pool.shutdown()
            torch_sync()

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        return config.global_config.image.supported_dims
        
    def assign_aspect_buckets(
        self,
        image_paths: List[str],
        buckets: List[Tuple[int, int]],
        max_aspect_ratio: float,
        batch_size: int = 256
    ) -> List[int]:
        if not hasattr(self, '_bucket_cache'):
            self._bucket_cache = {}
        bucket_indices = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_indices = []
            to_process = [path for path in batch_paths if path not in self._bucket_cache]
            if to_process:
                chunks = [to_process[i:i + 32] for i in range(0, len(to_process), 32)]
                with ThreadPoolExecutor(max_workers=self.num_proc) as pool:
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
                    for path, future in zip(to_process, all_futures):
                        try:
                            bucket_idx = future.result()
                            self._bucket_cache[path] = bucket_idx
                        except Exception as e:
                            logger.error(f"Error in bucket assignment: {str(e)}")
                            self._bucket_cache[path] = 0
            batch_indices = [self._bucket_cache.get(path, 0) for path in batch_paths]
            bucket_indices.extend(batch_indices)
        return bucket_indices
        
    def _assign_single_bucket(
        self,
        img_path: str,
        buckets: List[Tuple[int, int]],
        max_aspect_ratio: float
    ) -> int:
        try:
            img = Image.open(img_path)
            w, h = img.size
            aspect_ratio = w / h
            img_area = w * h
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
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            chunk_bytes = int(total_memory * self.max_chunk_memory)
        else:
            chunk_bytes = self.stream_buffer_size
        with open(path, 'wb') as f:
            for tensor_dict in self._chunk_tensor_dict(data, chunk_bytes):
                torch.save(tensor_dict, f, _use_new_zipfile_serialization=True)

    def _chunk_tensor_dict(self, data: Dict, chunk_bytes: int):
        current_bytes = 0
        current_chunk = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensor_bytes = value.nelement() * value.element_size()
                if tensor_bytes > chunk_bytes:
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
