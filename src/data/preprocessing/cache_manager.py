"""High-performance cache management with extreme speedups."""
import multiprocessing as mp
import logging
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
from src.core.logging.logging import setup_logging

# Initialize logger with core logging system
logger = setup_logging(__name__)

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
from src.data.utils.paths import convert_windows_path

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Enhanced cache statistics with detailed tracking."""
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
    
    def _log_action(self, operation: str, context: Dict[str, Any] = None):
        """Log operation with detailed context using core logging."""
        timestamp = time.time()
        
        # Track memory if available
        memory_info = {}
        if torch.cuda.is_available():
            memory_info.update({
                'cuda_memory_allocated': torch.cuda.memory_allocated(),
                'cuda_memory_reserved': torch.cuda.memory_reserved(),
                'cuda_max_memory': torch.cuda.max_memory_allocated()
            })
        
        # Add operation to history
        action_id = f"{operation}_{timestamp}"
        self.action_history[action_id] = {
            'operation': operation,
            'timestamp': timestamp,
            'memory': memory_info,
            'context': context or {}
        }
        
        # Log with enhanced context
        self.logger.debug(f"Cache operation: {operation}", extra={
            'operation': operation,
            'memory_stats': memory_info,
            'context': context
        })

    def _handle_error(self, operation: str, error: Exception, context: Dict[str, Any] = None):
        """Handle errors with detailed logging."""
        error_context = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc()
        }
        if context:
            error_context.update(context)
            
        self.logger.error(f"Cache operation failed: {operation}", extra=error_context)
        
        # Track error in history
        self._log_action(f"error_{operation}", error_context)
        
        # Update relevant stats
        self.stats.failed_items += 1
        if isinstance(error, torch.cuda.OutOfMemoryError):
            self.stats.gpu_oom_events += 1
        elif isinstance(error, IOError):
            self.stats.io_errors += 1

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
        model_dtypes: Optional[ModelWeightDtypes] = None,
        device: Optional[torch.device] = None
    ):
        # Wait for any pending CUDA operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Get or default device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Track initialization state
        self.initialized = False
        
        try:
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
            self.action_history = {}
            self.logger = logger
            self.memory_stats = {
                'peak_allocated': 0,
                'total_allocated': 0,
                'num_allocations': 0,
                'oom_events': 0
            }
            # Setup directories
            self.text_dir = self.cache_dir / "text"
            self.image_dir = self.cache_dir / "image"
            for directory in [self.text_dir, self.image_dir]:
                directory.mkdir(exist_ok=True)
                
            self.index_path = self.cache_dir / "cache_index.json"
            
            # Load cache index with retry logic
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            for attempt in range(max_retries):
                try:
                    self.cache_index = self._load_cache_index()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Cache index load attempt {attempt + 1} failed: {str(e)}, retrying...")
                        time.sleep(retry_delay)
                    else:
                        logger.error("Failed to load cache index after all retries")
                        self.cache_index = {"files": {}, "chunks": {}}

            # Verify initialization
            self.initialized = True
            logger.info(f"Cache manager initialized on device {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {str(e)}")
            raise

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


    def _load_cache_index(self) -> Dict:
        """Load and validate cache index with optimized performance."""
        try:
            if not self.initialized:
                return {"files": {}, "chunks": {}}

            # Fast parallel file scanning using sets
            with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                image_future = executor.submit(lambda: {p.stem for p in self.image_dir.glob("*.pt")})
                text_future = executor.submit(lambda: {p.stem for p in self.text_dir.glob("*.pt")})
                image_stems = image_future.result()
                text_stems = text_future.result()

            # Pre-compute file paths for O(1) lookup
            image_files = {
                stem: self.image_dir / f"{stem}.pt"
                for stem in image_stems
            }
            text_files = {
                stem: self.text_dir / f"{stem}.pt"
                for stem in text_stems & image_stems  # Intersection for only valid pairs
            }

            # Load index with error handling
            try:
                with open(self.index_path, 'r') as f:
                    index_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                index_data = {"files": {}, "chunks": {}}

            # Fast batch processing of entries
            valid_files = {}
            existing_stems = {Path(p).stem for p in index_data.get("files", {})}
            new_stems = image_stems - existing_stems

            # Process existing entries in batches
            batch_size = 1000
            for batch_start in range(0, len(index_data.get("files", {})), batch_size):
                batch_items = list(index_data["files"].items())[batch_start:batch_start + batch_size]
                
                for file_path, file_info in batch_items:
                    base_name = Path(file_path).stem
                    if base_name in image_stems:
                        file_info.update({
                            "latent_path": str(image_files[base_name]),
                            "type": "image",
                            "text_path": str(text_files[base_name]) if base_name in text_stems else None,
                            "text_type": "text" if base_name in text_stems else None
                        })
                        valid_files[file_path] = file_info

            # Process new entries in parallel
            def process_new_stem(stem):
                original_path = str(image_files[stem]).replace(".pt", ".jpg")
                return original_path, {
                    "base_name": stem,
                    "latent_path": str(image_files[stem]),
                    "type": "image",
                    "timestamp": time.time(),
                    "text_path": str(text_files[stem]) if stem in text_stems else None,
                    "text_type": "text" if stem in text_stems else None
                }

            with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                new_entries = dict(executor.map(process_new_stem, new_stems))
                valid_files.update(new_entries)

            index_data["files"] = valid_files
            self._save_cache_index(index_data)
            return index_data

        except Exception as e:
            logger.error(f"Failed to load cache index: {str(e)}")
            return {"files": {}, "chunks": {}}
        
    def validate_cache_index(self) -> Tuple[List[str], List[str]]:
        """Validate cache index by ensuring all referenced files exist and are valid.
        
        Returns:
            Tuple containing:
            - List of paths missing text embeddings
            - List of paths missing or invalid latent files
        """
        valid_files = {}
        missing_text = set()
        missing_latents = set()
        
        # Parallel file scanning using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
            latent_future = executor.submit(lambda: {p.stem for p in self.image_dir.glob("*.pt")})
            text_future = executor.submit(lambda: {p.stem for p in self.text_dir.glob("*.pt")})
            latent_stems = latent_future.result()
            text_stems = text_future.result()

        # Fast lookup preparation
        index_stems = {Path(p).stem for p in self.cache_index.get("files", {})}
        needed_stems = index_stems & latent_stems
        
        # Pre-compute file paths
        latent_files = {
            stem: self.image_dir / f"{stem}.pt"
            for stem in needed_stems
        }

        def validate_single_file(item: Tuple[str, dict]) -> Tuple[str, dict, bool, bool]:
            """Validate a single cache entry with optimized checks.
            
            Returns:
                Tuple of (file_path, file_info, is_valid, needs_text)
            """
            file_path, file_info = item
            base_name = Path(file_path).stem
            
            if base_name not in latent_stems:
                return file_path, None, False, False
                
            latent_path = latent_files.get(base_name)
            if not latent_path:
                return file_path, None, False, False
                
            try:
                # Fast validation using memory mapping if available
                latent_data = torch.load(latent_path, map_location='cpu')
                if not (latent_data.get("latent") and 
                    latent_data.get("metadata", {}).get("timestamp")):
                    return file_path, None, False, False
                    
                file_info["latent_path"] = str(latent_path)
                file_info["timestamp"] = latent_data["metadata"]["timestamp"]
                
                needs_text = base_name not in text_stems
                if not needs_text:
                    file_info["text_path"] = str(self.text_dir / f"{base_name}.pt")
                    
                return file_path, file_info, True, needs_text
                
            except Exception as e:
                logger.warning(f"Invalid latent file for {file_path}: {e}")
                return file_path, None, False, False

        # Parallel validation of files
        batch_size = 1000
        cache_items = list(self.cache_index.get("files", {}).items())
        
        for batch_start in range(0, len(cache_items), batch_size):
            batch = cache_items[batch_start:batch_start + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                results = list(executor.map(validate_single_file, batch))
                
                # Process results in batch
                for file_path, file_info, is_valid, needs_text in results:
                    if is_valid:
                        valid_files[file_path] = file_info
                        if needs_text:
                            missing_text.add(file_path)
                    else:
                        missing_latents.add(file_path)

        # Batch update index
        self.cache_index["files"] = valid_files
        self._save_cache_index()
        
        return list(missing_text), list(missing_latents)

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


    def save_preprocessed_data(
        self,
        latent_data: Optional[Dict[str, torch.Tensor]],
        text_embeddings: Optional[Dict[str, torch.Tensor]],
        metadata: Dict,
        file_path: Union[str, Path],
        caption: Optional[str] = None
    ) -> bool:
        """Save preprocessed data with optimized parallel writes."""
        if latent_data is None and text_embeddings is None:
            logger.error(f"Both latent_data and text_embeddings are None for {file_path}")
            return False

        try:
            if self.enable_memory_tracking:
                self._track_memory("save_start")
                
            base_name = Path(file_path).stem
            str_path = str(file_path)
            
            # Prepare metadata once
            metadata = metadata.copy()
            metadata["timestamp"] = time.time()
            if caption is not None:
                metadata["caption"] = caption

            # Prepare file info
            file_info = {
                "base_name": base_name,
                "timestamp": metadata["timestamp"],
                **self.cache_index["files"].get(str_path, {})
            }

            def save_tensor_file(data: Dict, path: Path, key: str) -> Optional[str]:
                """Save tensor file with error handling."""
                try:
                    torch.save(data, path)
                    return str(path)
                except Exception as e:
                    logger.error(f"Failed to save {key} data: {e}")
                    return None

            # Parallel saving of latent and text data
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                if latent_data is not None:
                    latent_path = self.image_dir / f"{base_name}.pt"
                    futures.append(
                        executor.submit(save_tensor_file, 
                            {"latent": latent_data, "metadata": metadata},
                            latent_path, "latent"
                        )
                    )
                    
                if text_embeddings is not None:
                    text_path = self.text_dir / f"{base_name}.pt"
                    futures.append(
                        executor.submit(save_tensor_file,
                            {"embeddings": text_embeddings, "metadata": metadata},
                            text_path, "text"
                        )
                    )

                # Process results and update file info
                for future, (data_type, path) in zip(futures, [
                    ("latent", self.image_dir / f"{base_name}.pt"),
                    ("text", self.text_dir / f"{base_name}.pt")
                ][:len(futures)]):
                    if saved_path := future.result():
                        if data_type == "latent":
                            file_info.update({
                                "latent_path": saved_path,
                                "type": "image"
                            })
                        else:
                            file_info.update({
                                "text_path": saved_path,
                                "text_type": "text"
                            })

            # Batch update index
            if file_info.get("latent_path") or file_info.get("text_path"):
                self.cache_index["files"][str_path] = file_info
                self._save_cache_index()
                return True
                
            return False

        except Exception as e:
            logger.error(f"Failed to save preprocessed data for {file_path}: {str(e)}")
            return False

    def has_cached_item(self, file_path: Union[str, Path]) -> bool:
        """Fast check for cached item existence."""
        str_path = str(file_path)
        file_info = self.cache_index["files"].get(str_path)
        if not file_info:
            return False

        # Fast existence check without loading data
        if "latent_path" in file_info:
            latent_exists = Path(file_info["latent_path"]).exists()
        else:
            latent_exists = False

        if "text_path" in file_info:
            text_exists = Path(file_info["text_path"]).exists()
        else:
            text_exists = False

        return latent_exists or text_exists

    def load_preprocessed_data(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Optimized loader using get_cached_item."""
        return self.get_cached_item(file_path)
        
    def get_cached_item(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached item with optimized memory and speed."""
        try:
            str_path = str(file_path)
            base_name = Path(str_path).stem

            # Fast path: check if entry exists
            file_info = self.cache_index["files"].get(str_path)
            if not file_info:
                # Quick lookup using pre-computed paths
                latent_path = self.image_dir / f"{base_name}.pt"
                text_path = self.text_dir / f"{base_name}.pt"
                
                if latent_path.exists() and text_path.exists():
                    file_info = {
                        "latent_path": str(latent_path),
                        "text_path": str(text_path),
                        "base_name": base_name,
                        "timestamp": time.time()
                    }
                    self.cache_index["files"][str_path] = file_info
                    self._save_cache_index()
                else:
                    self.stats.cache_misses += 1
                    return None

            # Parallel loading of latent and text data
            result = {}
            metadata = {}

            def load_tensor_file(path: Path, target_key: str):
                if path.exists():
                    data = torch.load(path, map_location='cpu')
                    if target_key == "latent":
                        return {"latent": data["latent"], "metadata": data["metadata"]}
                    else:
                        return {"text_embeddings": data["embeddings"], "metadata": data["metadata"]}
                return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                if "latent_path" in file_info:
                    futures.append(executor.submit(load_tensor_file, Path(file_info["latent_path"]), "latent"))
                if "text_path" in file_info:
                    futures.append(executor.submit(load_tensor_file, Path(file_info["text_path"]), "text_embeddings"))

                # Process results as they complete
                for future in futures:
                    data = future.result()
                    if data:
                        result.update({k: v for k, v in data.items() if k != "metadata"})
                        metadata.update(data["metadata"])

            if not result:
                self.stats.cache_misses += 1
                return None

            # Move tensors to device if specified
            if device is not None:
                for component in result.values():
                    if isinstance(component, dict):
                        for k, v in component.items():
                            if isinstance(v, torch.Tensor):
                                component[k] = v.to(device, non_blocking=True)

            result["metadata"] = metadata
            self.stats.cache_hits += 1
            return result

        except Exception as e:
            logger.error(f"Error retrieving cached item: {str(e)}")
            self.stats.failed_items += 1
            return None


    def load_text_embeddings(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load text embeddings and metadata from cache with optimized memory handling.
        
        Args:
            file_path: Path to original image file
            device: Optional device to load tensors onto
                
        Returns:
            Dictionary containing embeddings and metadata if available, None otherwise
        """
        try:
            if self.enable_memory_tracking:
                self._track_memory("load_text_start")

            # Fast path: check cache index first
            str_path = str(file_path)
            file_info = self.cache_index["files"].get(str_path)
            if not file_info or "text_path" not in file_info:
                self.stats.cache_misses += 1
                return None

            text_path = Path(file_info["text_path"])
            if not text_path.exists():
                # Update cache index to remove invalid entry
                if "text_path" in file_info:
                    del file_info["text_path"]
                    self._save_cache_index()
                self.stats.corrupted_items += 1
                return None

            # Optimize device placement
            target_device = device or 'cpu'
            use_non_blocking = (
                torch.cuda.is_available() and 
                isinstance(target_device, torch.device) and 
                target_device.type == 'cuda'
            )

            # Load with memory optimization
            with torch.cuda.amp.autocast(enabled=use_non_blocking):
                text_data = torch.load(text_path, map_location='cpu')
                
                # Validate loaded data
                if not isinstance(text_data, dict) or 'embeddings' not in text_data:
                    logger.warning(f"Invalid text embeddings format in {text_path}")
                    self.stats.corrupted_items += 1
                    return None

                # Optimize tensor transfer
                if target_device != 'cpu':
                    embeddings = text_data['embeddings']
                    if isinstance(embeddings, dict):
                        for k, v in embeddings.items():
                            if isinstance(v, torch.Tensor):
                                embeddings[k] = v.to(
                                    target_device, 
                                    non_blocking=use_non_blocking
                                )
                    elif isinstance(embeddings, torch.Tensor):
                        text_data['embeddings'] = embeddings.to(
                            target_device,
                            non_blocking=use_non_blocking
                        )

                # Record stream if using CUDA
                if use_non_blocking and self.stream is not None:
                    for v in text_data['embeddings'].values():
                        if isinstance(v, torch.Tensor):
                            v.record_stream(self.stream)

            self.stats.cache_hits += 1
            
            if self.enable_memory_tracking:
                self._track_memory("load_text_end")
                
            return text_data

        except Exception as e:
            logger.error(f"Error loading text embeddings for {file_path}: {str(e)}")
            self.stats.failed_items += 1
            return None
        
        


    def _verify_cache_entry(self, file_path: str, file_info: Dict) -> bool:
        """Verify cache entry validity with fast checks."""
        try:
            # Fast path: check required fields using set operation
            required_fields = {"base_name", "timestamp"}
            if not required_fields.issubset(file_info.keys()):
                return False
            
            # Batch existence check for both paths
            paths_to_check = []
            if "latent_path" in file_info:
                paths_to_check.append(Path(file_info["latent_path"]))
            if "text_path" in file_info:
                paths_to_check.append(Path(file_info["text_path"]))
                
            # Fast existence check using any()
            return all(path.exists() for path in paths_to_check)
                
        except Exception as e:
            logger.warning(f"Error verifying cache entry {file_path}: {str(e)}")
            return False

    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Perform comprehensive cache validation with parallel processing."""
        validation_stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'invalid_metadata': 0,
            'errors': []
        }
        
        def validate_single_entry(item: Tuple[str, Dict]) -> Dict[str, Any]:
            """Validate a single cache entry with optimized checks."""
            file_path, file_info = item
            result = {
                'is_valid': False,
                'error_type': None,
                'error_msg': None
            }
            
            try:
                # Validate latent file
                if "latent_path" in file_info:
                    latent_path = Path(file_info["latent_path"])
                    if not latent_path.exists():
                        result['error_type'] = 'missing'
                        return result
                    
                    # Fast tensor validation using memory mapping
                    try:
                        data = torch.load(latent_path, map_location='cpu')
                        if not isinstance(data, dict) or "latent" not in data:
                            result['error_type'] = 'corrupted'
                            return result
                            
                        # Quick metadata check
                        if "metadata" not in data:
                            result['error_type'] = 'invalid_metadata'
                            return result
                            
                    except Exception as e:
                        result.update({
                            'error_type': 'corrupted',
                            'error_msg': str(e)
                        })
                        return result
                        
                result['is_valid'] = True
                return result
                
            except Exception as e:
                result.update({
                    'error_type': 'error',
                    'error_msg': str(e)
                })
                return result

        try:
            cache_items = list(self.cache_index.get("files", {}).items())
            validation_stats['total_entries'] = len(cache_items)
            
            # Process in parallel batches
            batch_size = 1000
            with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                for batch_start in range(0, len(cache_items), batch_size):
                    batch = cache_items[batch_start:batch_start + batch_size]
                    
                    # Process batch in parallel
                    results = list(executor.map(validate_single_entry, batch))
                    
                    # Update statistics
                    for result, (file_path, _) in zip(results, batch):
                        if result['is_valid']:
                            validation_stats['valid_entries'] += 1
                        else:
                            error_type = result['error_type']
                            if error_type == 'missing':
                                validation_stats['missing_files'] += 1
                            elif error_type == 'corrupted':
                                validation_stats['corrupted_files'] += 1
                            elif error_type == 'invalid_metadata':
                                validation_stats['invalid_metadata'] += 1
                                
                            if result['error_msg']:
                                validation_stats['errors'].append({
                                    'file': file_path,
                                    'error': result['error_msg']
                                })

            logger.info(f"Cache validation complete: {validation_stats}")
            return validation_stats
            
        except Exception as e:
            logger.error(f"Cache validation failed: {str(e)}")
            raise PreprocessingError("Cache validation failed", context={
                'error': str(e),
                'stats': validation_stats
            })
    
    def _save_chunked_tensor(self, data: Dict, path: Path) -> None:
        """Save tensor data in optimized chunks with parallel processing."""
        try:
            # Calculate optimal chunk size
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                chunk_bytes = int(total_memory * self.max_chunk_memory)
                # Ensure chunk size is aligned with GPU memory
                chunk_bytes = (chunk_bytes // 256) * 256  # Align to 256 bytes
            else:
                chunk_bytes = self.stream_buffer_size

            # Prepare buffer for efficient writing
            buffer_size = min(chunk_bytes, 64 * 1024 * 1024)  # 64MB max buffer
            
            # Use temporary file for atomic writes
            temp_path = path.with_suffix('.tmp')
            
            with open(temp_path, 'wb', buffering=buffer_size) as f:
                # Process chunks with memory tracking
                for chunk_idx, tensor_dict in enumerate(self._chunk_tensor_dict(data, chunk_bytes)):
                    if self.enable_memory_tracking:
                        self._track_memory(f"save_chunk_{chunk_idx}")
                        
                    # Optimize serialization
                    torch.save(
                        tensor_dict,
                        f,
                        _use_new_zipfile_serialization=True,
                        pickle_protocol=5  # Use latest protocol for better performance
                    )
                    
                    # Force flush every N chunks to prevent memory buildup
                    if chunk_idx % 10 == 0:
                        f.flush()
                        
            # Atomic rename
            temp_path.rename(path)
            
        except Exception as e:
            # Cleanup on error
            if temp_path.exists():
                temp_path.unlink()
            raise PreprocessingError("Failed to save chunked tensor", context={
                'path': str(path),
                'error': str(e)
            })

    def _chunk_tensor_dict(self, data: Dict, chunk_bytes: int):
        """Generate optimized tensor chunks with memory efficiency."""
        try:
            current_bytes = 0
            current_chunk = {}
            metadata = data.get("metadata", {})
            
            # Pre-calculate tensor sizes
            tensor_sizes = {
                key: value.nelement() * value.element_size()
                for key, value in data.items()
                if isinstance(value, torch.Tensor)
            }
            
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    tensor_bytes = tensor_sizes[key]
                    
                    # Handle large tensors
                    if tensor_bytes > chunk_bytes:
                        # Calculate optimal split size
                        split_size = chunk_bytes // value.element_size()
                        split_size = (split_size // 256) * 256  # Align to 256 bytes
                        
                        # Use efficient splitting
                        splits = value.split(split_size)
                        for i, split in enumerate(splits):
                            # Create memory-efficient chunk
                            chunk = {
                                f"{key}_chunk_{i}": split.contiguous(),  # Ensure contiguous memory
                                "metadata": {
                                    **metadata,
                                    "chunk_info": {
                                        "total_chunks": len(splits),
                                        "chunk_index": i,
                                        "original_key": key
                                    }
                                }
                            }
                            yield chunk
                            
                            # Clear reference to help GC
                            del split
                            
                    else:
                        # Handle normal tensors
                        if current_bytes + tensor_bytes > chunk_bytes:
                            if current_chunk:
                                yield current_chunk
                            current_chunk = {}
                            current_bytes = 0
                            
                        # Store tensor efficiently
                        current_chunk[key] = value.contiguous()
                        current_bytes += tensor_bytes
                else:
                    # Handle non-tensor data
                    current_chunk[key] = value
                    
            # Yield final chunk if not empty
            if current_chunk:
                yield current_chunk
                
        except Exception as e:
            logger.error(f"Error chunking tensor dict: {str(e)}")
            raise PreprocessingError("Failed to chunk tensor dict", context={
                'error': str(e)
            })
    
    def sync_device(self):
        """Ensure all device operations are complete with optimized synchronization."""
        try:
            if not torch.cuda.is_available():
                return

            # Track sync start time for performance monitoring
            sync_start = time.perf_counter()

            # Optimize synchronization order
            if hasattr(self, 'stream') and self.stream is not None:
                # Stream-specific sync first
                self.stream.synchronize()
                
                # Record stream event for better tracking
                if hasattr(self, 'sync_event'):
                    self.sync_event.record()
            
            # Global CUDA sync
            torch.cuda.synchronize()
            
            if self.enable_memory_tracking:
                sync_time = time.perf_counter() - sync_start
                self._track_memory("device_sync", {
                    'sync_duration': sync_time,
                    'cuda_memory': torch.cuda.memory_allocated(),
                    'cuda_max_memory': torch.cuda.max_memory_allocated()
                })
                
        except Exception as e:
            logger.error(f"Device synchronization failed: {str(e)}")
            raise PreprocessingError("Device sync failed", context={
                'error': str(e),
                'cuda_available': torch.cuda.is_available(),
                'has_stream': hasattr(self, 'stream')
            })

    def clear_cache(self, remove_files: bool = True):
        """Clear cache with optimized file removal and memory cleanup."""
        try:
            if self.enable_memory_tracking:
                self._track_memory("clear_cache_start")

            # Ensure device sync before cleanup
            self.sync_device()
            
            if remove_files:
                import shutil
                
                def remove_directory(dir_path: Path) -> bool:
                    """Safely remove directory with error handling."""
                    try:
                        if dir_path.exists():
                            shutil.rmtree(dir_path)
                        dir_path.mkdir(parents=True)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to remove directory {dir_path}: {e}")
                        return False

                # Parallel directory removal
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(
                        remove_directory,
                        [self.text_dir, self.image_dir]
                    ))
                    
                    if not all(results):
                        raise PreprocessingError("Failed to remove cache directories")

            # Reset cache state
            self.cache_index = {"files": {}, "chunks": {}}
            self._save_cache_index()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(self, 'stream'):
                    self.stream.synchronize()
            
            # Reset statistics
            self.stats = CacheStats()
            
            # Reinitialize memory tracking
            if self.enable_memory_tracking:
                self._init_memory_tracking()
                self._track_memory("clear_cache_end")
                
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise PreprocessingError("Cache clear failed", context={
                'error': str(e),
                'remove_files': remove_files,
                'cuda_available': torch.cuda.is_available()
            })
        
    def __enter__(self):
        """Initialize context with optimized setup."""
        try:
            # Track context entry for debugging
            if self.enable_memory_tracking:
                self._track_memory("context_enter")
                
            # Ensure clean CUDA state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if hasattr(self, 'stream'):
                    self.stream.synchronize()
                    
            return self
            
        except Exception as e:
            logger.error(f"Error entering cache manager context: {str(e)}")
            raise PreprocessingError("Context entry failed", context={
                'error': str(e),
                'cuda_available': torch.cuda.is_available()
            })

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources with comprehensive error handling."""
        try:
            # Track context exit
            if self.enable_memory_tracking:
                self._track_memory("context_exit")
                
            # Save cache index with retry logic
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    self._save_cache_index()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to save cache index after {max_retries} attempts: {e}")
                        raise
                    time.sleep(retry_delay)
                    
        except Exception as e:
            logger.error(f"Error in cache manager cleanup: {str(e)}")
            if exc_type is None:
                raise PreprocessingError("Context exit failed", context={
                    'error': str(e)
                })
                
        finally:
            try:
                # Ensure CUDA cleanup
                if torch.cuda.is_available():
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Synchronize stream if exists
                    if hasattr(self, 'stream') and self.stream is not None:
                        self.stream.synchronize()
                    
                    # Final synchronization
                    torch.cuda.synchronize()
                    
                # Clean up any temporary resources
                if hasattr(self, '_temp_files'):
                    for temp_file in self._temp_files:
                        try:
                            if temp_file.exists():
                                temp_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
                            
            except Exception as e:
                logger.error(f"Error in final cleanup: {str(e)}")
                if exc_type is None:
                    raise PreprocessingError("Final cleanup failed", context={
                        'error': str(e)
                    })
