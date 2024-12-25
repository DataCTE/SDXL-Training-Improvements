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
from src.data.preprocessing.exceptions import PreprocessingError


from tqdm.auto import tqdm
from contextlib import nullcontext

from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.data.utils.tensor_utils import validate_tensor
from src.core.memory.tensor import (
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    device_equals,
    replace_tensors_,
    torch_sync
)
import logging
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
        max_chunk_memory: float = 0.2,
        model_dtypes: Optional[ModelWeightDtypes] = None
    ):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        self.cache_dir = Path(convert_windows_path(cache_dir, make_absolute=True))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if model_dtypes is None:
            self.model_dtypes = ModelWeightDtypes(
                train_dtype=DataType.FLOAT_32,
                fallback_train_dtype=DataType.FLOAT_32,
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
        else:
            self.model_dtypes = model_dtypes
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
        self.index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict:
        """Load and validate the cache index, scanning for existing files if needed."""
        try:
            index_data = {"files": {}, "chunks": {}}
            
            # First pass: Scan and process image files
            image_files = {p.stem: p for p in self.image_dir.glob("*.pt")}
            
            # Try loading existing index
            if self.index_path.exists():
                try:
                    with open(self.index_path, 'r') as f:
                        index_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning("Invalid or missing cache index file, starting fresh")
                    index_data = {"files": {}, "chunks": {}}
            else:
                logger.info("Creating new cache index")
                    
            # First validate image entries
            valid_files = {}
            for file_path, file_info in index_data.get("files", {}).items():
                base_name = Path(file_path).stem
                
                # Check for existing image file
                image_path = image_files.get(base_name)
                if image_path and image_path.exists():
                    file_info["latent_path"] = str(image_path)
                    file_info["type"] = "image"
                    valid_files[file_path] = file_info
            
            # Add any new image files found on disk
            for base_name, image_path in image_files.items():
                matching_entries = [p for p, info in valid_files.items() 
                                if Path(info.get("latent_path", "")).stem.replace("__image", "") == base_name]
                if not matching_entries:
                    file_info = {
                        "base_name": base_name,
                        "latent_path": str(image_path),
                        "type": "image",
                        "timestamp": time.time()
                    }
                    valid_files[str(image_path)] = file_info
            
            # Second pass: Process text files
            text_files = {p.stem: p for p in self.text_dir.glob("*.pt")}
            
            # Update existing entries with text information
            for file_path, file_info in valid_files.items():
                base_name = Path(file_path).stem
                text_path = text_files.get(base_name)
                if text_path and text_path.exists():
                    file_info["text_path"] = str(text_path)
                    file_info["text_type"] = "text"
            
            # Add any files found on disk but not in index
            for base_name, latent_path in image_files.items():
                matching_entries = [p for p, info in valid_files.items() 
                                 if Path(info.get("latent_path", "")).stem == base_name]
                if not matching_entries:
                    text_path = text_files.get(base_name)
                    # Reconstruct original path from base name
                    original_path = str(latent_path).replace(".pt", ".jpg")  # Assume .jpg as default
                    file_info = {
                        "base_name": base_name,
                        "latent_path": str(latent_path),
                        "timestamp": time.time()
                    }
                    if text_path and text_path.exists():
                        file_info["text_path"] = str(text_path)
                    # Use reconstructed original path as key
                    valid_files[original_path] = file_info
                    logger.debug(f"Added cache entry for {original_path}")
            
            index_data["files"] = valid_files
            # Save validated index
            self._save_cache_index(index_data)
            return index_data
            
        except Exception as e:
            logger.error(f"Failed to load cache index: {str(e)}")
            return {"files": {}, "chunks": {}}

    def _save_cache_index(self, index_data: Optional[Dict] = None) -> None:
        """Save the cache index to disk.
        
        Args:
            index_data: Optional index data to save, uses self.cache_index if None
        """
        try:
            data = index_data if index_data is not None else self.cache_index
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2)  # Pretty print for readability
            logger.info(f"Cache index saved to {self.index_path} with {len(data.get('files', {}))} entries")
        except Exception as e:
            logger.error(f"Failed to save cache index: {str(e)}")

    def validate_cache_index(self) -> Tuple[List[str], List[str]]:
        """Validate cache index by ensuring all referenced files exist and are valid.
        
        Returns:
            Tuple containing:
            - List of paths missing text embeddings
            - List of paths missing or invalid latent files
        """
        valid_files = {}
        missing_text = []
        missing_latents = []
        
        # First scan actual files on disk
        latent_files = {p.stem: p for p in self.image_dir.glob("*.pt")}
        text_files = {p.stem: p for p in self.text_dir.glob("*.pt")}
        
        # Validate each entry in the index
        for file_path, file_info in self.cache_index.get("files", {}).items():
            base_name = Path(file_path).stem
            
            # Check latent file and validate if it exists
            latent_path = latent_files.get(base_name)
            if latent_path and latent_path.exists():
                try:
                    # Quick validation without loading full tensor
                    latent_data = torch.load(latent_path, map_location='cpu')
                    if "latent" in latent_data and "metadata" in latent_data:
                        # Check if metadata contains timestamp
                        if "timestamp" in latent_data["metadata"]:
                            file_info["latent_path"] = str(latent_path)
                            file_info["timestamp"] = latent_data["metadata"]["timestamp"]
                            continue
                except Exception as e:
                    logger.warning(f"Invalid latent file for {file_path}: {e}")
                    
            missing_latents.append(file_path)
            logger.warning(f"Missing or invalid latent file for: {file_path}")
            continue
                
            # Check text embeddings file  
            text_path = text_files.get(base_name)
            if text_path and text_path.exists():
                file_info["text_path"] = str(text_path)
            else:
                missing_text.append(file_path)
                logger.warning(f"Missing text embeddings file for: {file_path}")
                
            # Keep entry if at least latents exist
            if latent_path:
                valid_files[file_path] = file_info

        # Update index with validated entries
        self.cache_index["files"] = valid_files
        self._save_cache_index()
        
        return missing_text, missing_latents

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
        file_path: Union[str, Path],
        caption: Optional[str] = None
    ) -> bool:
        if latent_data is None and text_embeddings is None:
            logger.error(f"Both latent_data and text_embeddings are None for {file_path}")
            return False
            
        if self.enable_memory_tracking:
            self._track_memory("save_start")
            
        base_name = Path(file_path).stem
        current_entry = self.cache_index["files"].get(str(file_path), {})
        metadata["timestamp"] = time.time()

        if caption is not None:
            metadata["caption"] = caption

        # Initialize or update the cache index entry
        file_info = current_entry.copy()
        file_info["base_name"] = base_name
        file_info["timestamp"] = metadata["timestamp"]

        try:
            # First handle image data if present
            if latent_data is not None:
                latent_path = self.image_dir / f"{base_name}.pt"
                torch.save({
                    "latent": latent_data,
                    "metadata": metadata
                }, latent_path)
                file_info["latent_path"] = str(latent_path)
                file_info["type"] = "image"
                # Save index after image processing
                self.cache_index["files"][str(file_path)] = file_info
                self._save_cache_index()

            # Then handle text data if present
            if text_embeddings is not None:
                text_path = self.text_dir / f"{base_name}.pt"
                torch.save({
                    "embeddings": text_embeddings,
                    "metadata": metadata
                }, text_path)
                file_info["text_path"] = str(text_path)
                file_info["text_type"] = "text"
                # Update index with text info
                self.cache_index["files"][str(file_path)] = file_info
                self._save_cache_index()

            return True
            
        except Exception as e:
            logger.error(f"Failed to save preprocessed data for {file_path}: {str(e)}")
            return False


    def _validate_tensor(self, tensor: torch.Tensor) -> bool:

        return validate_tensor(
            tensor,
            expected_dims=4,
            enable_memory_tracking=self.enable_memory_tracking,
            memory_stats=self.memory_stats if hasattr(self, 'memory_stats') else None
        )

    def has_cached_item(self, file_path: Union[str, Path]) -> bool:
        str_path = str(file_path)
        file_info = self.cache_index["files"].get(str_path)
        if not file_info:
            return None

        result = {}
        metadata = {}

        # Load latent data
        if "latent_path" in file_info:
            latent_path = Path(file_info["latent_path"])
            if latent_path.exists():
                latent_data = torch.load(latent_path, map_location='cpu')
                result["latent"] = latent_data["latent"]
                metadata.update(latent_data["metadata"])
            else:
                logger.warning(f"Latent file {latent_path} does not exist.")
                return None

        # Load text embeddings
        if "text_path" in file_info:
            text_path = Path(file_info["text_path"])
            if text_path.exists():
                text_data = torch.load(text_path, map_location='cpu')
                result["text_embeddings"] = text_data["embeddings"]
                metadata.update(text_data["metadata"])
            else:
                logger.warning(f"Text embeddings file {text_path} does not exist.")
                return None

        if not result:
            return None

        result["metadata"] = metadata
        return result
    
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
            result = {}
            metadata = {}
            if "latent_path" in file_info:
                latent_path = Path(file_info["latent_path"])
                if latent_path.exists():
                    latent_data = torch.load(latent_path, map_location='cpu')
                    result["latent"] = latent_data["latent"]
                    metadata.update(latent_data["metadata"])
                    if device is not None:
                        for k, v in result["latent"].items():
                            if isinstance(v, torch.Tensor):
                                result["latent"][k] = v.to(device, non_blocking=True)
                else:
                    self.stats.corrupted_items += 1
                    return None

            if "text_path" in file_info:
                text_path = Path(file_info["text_path"])
                if text_path.exists():
                    text_data = torch.load(text_path, map_location='cpu')
                    result["text_embeddings"] = text_data["embeddings"]
                    metadata.update(text_data["metadata"])
                    if device is not None:
                        for k, v in result["text_embeddings"].items():
                            if isinstance(v, torch.Tensor):
                                result["text_embeddings"][k] = v.to(device, non_blocking=True)
                else:
                    self.stats.corrupted_items += 1
                    return None

            if not result:
                self.stats.cache_misses += 1
                return None

            self.stats.cache_hits += 1
            result["metadata"] = metadata
            return result
        except Exception as e:
            logger.error(f"Error retrieving cached item: {str(e)}", exc_info=True)
            self.stats.failed_items += 1
            return None

    def load_text_embeddings(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load text embeddings and metadata from cache.
        
        Args:
            file_path: Path to original image file
            device: Optional device to load tensors onto
            
        Returns:
            Dictionary containing embeddings and metadata if available, None otherwise
        """
        try:
            file_info = self.cache_index["files"].get(str(file_path))
            if not file_info or "text_path" not in file_info:
                return None
            text_path = Path(file_info["text_path"])
            if not text_path.exists():
                return None
            text_data = torch.load(text_path, map_location=device or 'cpu')
            return text_data  # Contains 'embeddings' and 'metadata'
        except Exception as e:
            logger.error(f"Error loading text embeddings for {file_path}: {str(e)}")
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
            torch_sync()

        
        
    def _assign_single_bucket(
        self,
        img_path: str,
        buckets: List[Tuple[int, int]],
        max_aspect_ratio: float
    ) -> int:
        try:
            # Support more image formats
            img = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure compatibility
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
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Perform comprehensive cache validation with detailed reporting."""
        validation_stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'invalid_metadata': 0,
            'errors': []
        }
        
        try:
            for file_path, file_info in self.cache_index.get("files", {}).items():
                validation_stats['total_entries'] += 1
                
                try:
                    # Validate latent file
                    if "latent_path" in file_info:
                        latent_path = Path(file_info["latent_path"])
                        if not latent_path.exists():
                            validation_stats['missing_files'] += 1
                            continue
                            
                        # Quick validation of tensor file
                        try:
                            data = torch.load(latent_path, map_location='cpu')
                            if not isinstance(data, dict) or "latent" not in data:
                                validation_stats['corrupted_files'] += 1
                                continue
                        except Exception as e:
                            validation_stats['corrupted_files'] += 1
                            validation_stats['errors'].append({
                                'file': str(latent_path),
                                'error': str(e)
                            })
                            continue
                            
                    # Validate metadata
                    if "metadata" not in data:
                        validation_stats['invalid_metadata'] += 1
                        continue
                        
                    validation_stats['valid_entries'] += 1
                    
                except Exception as e:
                    logger.error(f"Error validating {file_path}: {str(e)}")
                    validation_stats['errors'].append({
                        'file': file_path,
                        'error': str(e)
                    })
                    
            logger.info(f"Cache validation complete: {validation_stats}")
            return validation_stats
            
        except Exception as e:
            logger.error(f"Cache validation failed: {str(e)}")
            raise PreprocessingError("Cache validation failed", context={
                'error': str(e),
                'stats': validation_stats
            })
