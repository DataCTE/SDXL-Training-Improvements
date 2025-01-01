"""Cache management for preprocessed latents with WSL support."""
from pathlib import Path
import json
import time
import torch
import threading
from typing import Dict, Optional, Union, Any, List, Tuple
from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path
import os

logger = get_logger(__name__)

class CacheManager:
    """Manages caching of preprocessed latents with WSL path support."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_cache_size: int = 10000,
        device: Optional[torch.device] = None
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
            max_cache_size: Maximum number of entries to keep
            device: Default device for loading tensors
        """
        # Convert and create cache directory
        self.cache_dir = convert_windows_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.latents_dir = self.cache_dir / "latents"
        self.metadata_dir = self.cache_dir / "metadata"
        self.latents_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        
        # Initialize cache index
        self._initialize_cache()
        logger.info(f"Cache initialized at {self.cache_dir}")
        
    def _initialize_cache(self):
        """Initialize or load cache state."""
        self.index_path = self.cache_dir / "cache_index.json"
        
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
            
    def _create_new_index(self):
        """Create new cache index structure."""
        self.cache_index = {
            "version": "1.0",
            "created_at": time.time(),
            "last_updated": time.time(),
            "entries": {},
            "stats": {
                "total_entries": 0,
                "latents_size": 0,
                "last_cleanup": None
            }
        }
        self._save_index()
        
    def _save_index(self) -> None:
        """Save cache index to disk with proper error handling."""
        try:
            # Create temporary file
            temp_path = self.index_path.with_suffix('.tmp')
            
            # Write to temporary file first
            with open(temp_path, 'w') as f:
                json.dump({
                    "version": self.cache_index["version"],
                    "created_at": self.cache_index["created_at"],
                    "last_updated": time.time(),
                    "entries": self.cache_index["entries"],
                    "stats": self.cache_index["stats"]
                }, f, indent=2)
            
            # Atomic rename to actual index file
            temp_path.replace(self.index_path)
            
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
            if temp_path.exists():
                temp_path.unlink()  # Clean up temp file if it exists
            raise

    def get_cache_key(self, path: Union[str, Path]) -> str:
        """Generate cache key from path using ultra-fast string operations."""
        # Convert to string only once
        path_str = str(path)
        
        # Find last directory separator
        last_sep = max(path_str.rfind('/'), path_str.rfind('\\'))
        if last_sep == -1:
            # No directory separator found, use whole path
            second_last_sep = -1
        else:
            # Find second-to-last separator
            second_last_sep = max(
                path_str.rfind('/', 0, last_sep),
                path_str.rfind('\\', 0, last_sep)
            )
        
        # Find extension dot
        dot_pos = path_str.rfind('.')
        
        # Extract parts directly using string slicing
        dir_name = path_str[second_last_sep + 1:last_sep]
        file_name = path_str[last_sep + 1:dot_pos]
        
        # Join with underscore using string concatenation
        return f"{dir_name}_{file_name}"

    def save_latents(
        self,
        tensors: Dict[str, torch.Tensor],
        original_path: Union[str, Path],
        metadata: Dict[str, Any]
    ) -> bool:
        """Optimized save latents with minimal IO operations."""
        cache_key = self.get_cache_key(original_path)
        tensors_path = self.latents_dir / f"{cache_key}.pt"
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        
        try:
            # Pre-process tensors before IO
            tensors_cpu = {k: v.cpu() for k, v in tensors.items()}
            
            # Prepare metadata once
            full_metadata = {
                "original_path": str(original_path),
                "created_at": time.time(),
                **metadata
            }
            
            # Batch IO operations
            torch.save(tensors_cpu, tensors_path)
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f)
            
            # Single lock acquisition for index update
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "tensors_path": str(tensors_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time(),
                    "is_valid": True,
                    "last_checked": time.time()
                }
                
                # Batch stats update
                self._update_stats()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return False
            
    def load_latents(
        self,
        path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Optimized latents loading with caching."""
        cache_key = self.get_cache_key(path)
        
        # Use cached entry status
        entry = self.cache_index["entries"].get(cache_key)
        if not entry:
            return None
        
        if "is_valid" in entry and not entry["is_valid"]:
            return None
        
        try:
            # Load tensors directly to target device with safe loading
            device = device or self.device
            tensors = torch.load(
                entry["tensors_path"], 
                map_location=device,
                weights_only=True  # Enable safe loading
            )
            
            # Load metadata with buffered IO
            with open(entry["metadata_path"]) as f:
                metadata = json.load(f)
            
            return {
                "pixel_values": tensors["pixel_values"],
                "prompt_embeds": tensors["prompt_embeds"],
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "original_size": metadata["original_size"],
                "crop_coords": metadata["crop_coords"],
                "target_size": metadata["target_size"],
                "text": metadata.get("text")
            }
            
        except Exception as e:
            # Mark entry as invalid on failure
            with self._lock:
                entry["is_valid"] = False
                entry["last_checked"] = time.time()
            logger.error(f"Failed to load from cache: {e}")
            return None
            
    def cleanup(self, max_age: Optional[float] = None):
        """Optimized cleanup with batch processing."""
        try:
            with self._lock:
                current_time = time.time()
                keys_to_remove = set()
                
                # Batch process entries
                for key, entry in self.cache_index["entries"].items():
                    if (max_age and (current_time - entry["created_at"]) > max_age) or \
                       (not Path(entry["tensors_path"]).exists() or not Path(entry["metadata_path"]).exists()):
                        keys_to_remove.add(key)
                
                if not keys_to_remove:
                    return
                
                # Batch delete files
                for key in keys_to_remove:
                    entry = self.cache_index["entries"][key]
                    try:
                        os.remove(entry["tensors_path"])
                        os.remove(entry["metadata_path"])
                    except OSError:
                        pass
                    del self.cache_index["entries"][key]
                
                # Single stats update
                self._update_stats()
                self.cache_index["stats"]["last_cleanup"] = current_time
                self._save_index()
                
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def _update_stats(self):
        """Batch update cache statistics."""
        self.cache_index["stats"].update({
            "total_entries": len(self.cache_index["entries"]),
            "latents_size": sum(
                os.path.getsize(str(p)) 
                for p in self.latents_dir.glob("*.pt")
            )
        })
        self.cache_index["last_updated"] = time.time()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def is_cached(self, path: Union[str, Path]) -> bool:
        """Check if path is already cached and valid.
        
        Optimized version that minimizes file system operations and uses
        in-memory checks where possible.
        """
        cache_key = self.get_cache_key(path)
        
        # Quick dictionary lookup first
        if cache_key not in self.cache_index["entries"]:
            return False
        
        entry = self.cache_index["entries"][cache_key]
        
        # Use cached validity status if available
        if "is_valid" in entry and time.time() - entry.get("last_checked", 0) < 60:
            return entry["is_valid"]
        
        # Check files exist
        tensors_path = Path(entry["tensors_path"])
        metadata_path = Path(entry["metadata_path"])
        
        is_valid = tensors_path.exists() and metadata_path.exists()
        
        # Cache the validity status
        with self._lock:
            entry["is_valid"] = is_valid
            entry["last_checked"] = time.time()
            
            if not is_valid:
                del self.cache_index["entries"][cache_key]
                self._save_index()
        
        return is_valid

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return {
            "total_entries": len(self.cache_index["entries"]),
            "cache_size_bytes": sum(
                os.path.getsize(str(p)) for p in self.latents_dir.glob("*.pt")
            ),
            "last_updated": self.cache_index["last_updated"],
            "last_cleanup": self.cache_index["stats"]["last_cleanup"]
        }

    def save_latents_batch(
        self,
        batch_tensors: List[Dict[str, torch.Tensor]],
        original_paths: List[Union[str, Path]],
        batch_metadata: List[Dict[str, Any]]
    ) -> List[bool]:
        """Highly optimized batch save using parallel processing."""
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        async def save_batch():
            # Pre-process all tensors to CPU in parallel
            with ThreadPoolExecutor() as pool:
                cpu_tensors = list(await asyncio.gather(*[
                    asyncio.get_event_loop().run_in_executor(
                        pool,
                        lambda t=t: {k: v.cpu() for k, v in t.items()}
                    )
                    for t in batch_tensors
                ]))
            
            # Prepare all paths and metadata
            cache_entries = [
                (
                    self.get_cache_key(path),
                    self.latents_dir / f"{self.get_cache_key(path)}.pt",
                    self.metadata_dir / f"{self.get_cache_key(path)}.json",
                    {
                        "original_path": str(path),
                        "created_at": time.time(),
                        **metadata
                    }
                )
                for path, metadata in zip(original_paths, batch_metadata)
            ]
            
            # Parallel save operations
            async def save_entry(tensors, entry_data, idx):
                key, tensor_path, metadata_path, metadata = entry_data
                try:
                    with ThreadPoolExecutor() as pool:
                        # Parallel file writes with safe saving
                        await asyncio.gather(
                            asyncio.get_event_loop().run_in_executor(
                                pool,
                                lambda: torch.save(
                                    tensors, 
                                    tensor_path,
                                    weights_only=True  # Enable safe saving
                                )
                            ),
                            asyncio.get_event_loop().run_in_executor(
                                pool,
                                lambda: json.dump(metadata, open(metadata_path, 'w'))
                            )
                        )
                    
                    return True, key, tensor_path, metadata_path
                except Exception as e:
                    logger.error(f"Failed to save batch entry {idx}: {e}")
                    return False, key, None, None
            
            # Process all entries in parallel
            results = await asyncio.gather(*[
                save_entry(t, e, i)
                for i, (t, e) in enumerate(zip(cpu_tensors, cache_entries))
            ])
            
            # Batch update index
            success_entries = [
                (key, str(tp), str(mp))
                for success, key, tp, mp in results
                if success
            ]
            
            if success_entries:
                with self._lock:
                    for key, tensor_path, metadata_path in success_entries:
                        self.cache_index["entries"][key] = {
                            "tensors_path": tensor_path,
                            "metadata_path": metadata_path,
                            "created_at": time.time(),
                            "is_valid": True,
                            "last_checked": time.time()
                        }
                    self._update_stats()
                    self._save_index()
            
            return [r[0] for r in results]
        
        return asyncio.run(save_batch())

    def load_tensors(self, cache_key: str, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
        """Load cached tensors with improved validation and error handling."""
        try:
            entry = self.cache_index["entries"].get(cache_key)
            if not entry:
                logger.debug(f"No cache entry found for key: {cache_key}")
                return None
            
            # Validate cache files exist
            tensor_path = Path(entry["tensors_path"])
            metadata_path = Path(entry["metadata_path"])
            
            if not tensor_path.exists() or not metadata_path.exists():
                logger.debug(f"Cache files missing for key: {cache_key}")
                self._invalidate_cache_entry(cache_key)
                return None
            
            # Load tensors and metadata
            device = device or self.device
            try:
                tensors = torch.load(
                    tensor_path,
                    map_location=device,
                    weights_only=True
                )
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    
                # Validate required tensor keys
                required_tensor_keys = {"pixel_values", "prompt_embeds", "pooled_prompt_embeds"}
                required_metadata_keys = {"original_size", "crop_coords", "target_size"}
                
                if not all(key in tensors for key in required_tensor_keys):
                    logger.warning(f"Cache entry {cache_key} missing required tensor keys")
                    self._invalidate_cache_entry(cache_key)
                    return None
                    
                if not all(key in metadata for key in required_metadata_keys):
                    logger.warning(f"Cache entry {cache_key} missing required metadata keys")
                    self._invalidate_cache_entry(cache_key)
                    return None
                    
                # Return combined dictionary with both tensors and metadata
                return {
                    **tensors,
                    "metadata": metadata
                }
                
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load cache entry {cache_key}: {str(e)}")
                self._invalidate_cache_entry(cache_key)
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error loading cache for {cache_key}: {str(e)}")
            return None
        
    def _invalidate_cache_entry(self, cache_key: str) -> None:
        """Invalidate and clean up a cache entry."""
        with self._lock:
            if cache_key in self.cache_index["entries"]:
                entry = self.cache_index["entries"][cache_key]
                # Try to clean up files
                try:
                    Path(entry["tensors_path"]).unlink(missing_ok=True)
                    Path(entry["metadata_path"]).unlink(missing_ok=True)
                except Exception:
                    pass
                # Remove from index
                del self.cache_index["entries"][cache_key]
                self._save_index()

    def save_tensors(self, tensors: Dict[str, torch.Tensor], cache_key: str) -> Tuple[bool, str, str]:
        """Save tensors with safe saving enabled."""
        try:
            # Generate paths
            tensors_path = self.latents_dir / f"{cache_key}.pt"
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            
            # Save tensors with weights_only=True
            torch.save(
                tensors,
                tensors_path,
                weights_only=True  # Safe saving mode
            )
            
            return True, str(tensors_path), str(metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to save tensors for {cache_key}: {str(e)}")
            return False, "", ""
