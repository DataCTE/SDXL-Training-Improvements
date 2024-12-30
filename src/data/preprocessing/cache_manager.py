"""Cache management for preprocessed latents with WSL support."""
from pathlib import Path
import json
import time
import torch
import threading
from typing import Dict, Optional, Union, Any, List
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
        """Generate cache key from path."""
        try:
            # Convert path to proper format
            path = convert_windows_path(path)
            # Use absolute path for more reliable keys
            path = path.resolve()
            # Create unique key from path
            return f"{path.parent.name}_{path.stem}"
        except Exception as e:
            logger.warning(f"Failed to generate optimal cache key for {path}, falling back to basic key")
            # Fallback to basic key if conversion fails
            path = Path(str(path))
            return path.stem

    def save_latents(
        self,
        tensors: Dict[str, torch.Tensor],
        original_path: Union[str, Path],
        metadata: Dict[str, Any]
    ) -> bool:
        """Save tensors and metadata to cache."""
        try:
            # Convert paths
            original_path = convert_windows_path(original_path)
            cache_key = self.get_cache_key(original_path)
            
            # Create paths for saving
            tensors_path = self.latents_dir / f"{cache_key}.pt"
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            
            # Move tensors to CPU and save
            tensors_to_save = {
                k: v.cpu() for k, v in tensors.items()
            }
            torch.save(tensors_to_save, tensors_path)
            
            # Save metadata with full path info
            full_metadata = {
                "original_path": str(original_path.resolve()),
                "created_at": time.time(),
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update index with thread safety
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "tensors_path": str(tensors_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time()
                }
                
                # Update stats
                self.cache_index["stats"]["total_entries"] = len(self.cache_index["entries"])
                self.cache_index["stats"]["latents_size"] = sum(
                    os.path.getsize(str(p)) for p in self.latents_dir.glob("*.pt")
                )
                self.cache_index["last_updated"] = time.time()
                
                # Save index file
                self._save_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}", exc_info=True)
            return False
            
    def load_latents(
        self,
        path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Load tensors and metadata from cache.
        
        Args:
            path: Original file path to load cache for
            device: Optional device to load tensors to
            
        Returns:
            Dict containing:
                - pixel_values: Tensor
                - prompt_embeds: Tensor
                - pooled_prompt_embeds: Tensor
                - original_size: Tuple[int, int]
                - crop_coords: Tuple[int, int]
                - target_size: Tuple[int, int]
                - text: str (optional)
        """
        try:
            cache_key = self.get_cache_key(path)
            
            if cache_key not in self.cache_index["entries"]:
                return None
                
            entry = self.cache_index["entries"][cache_key]
            tensors_path = Path(entry["tensors_path"])
            metadata_path = Path(entry["metadata_path"])
            
            if not tensors_path.exists() or not metadata_path.exists():
                # Clean up invalid entry
                with self._lock:
                    del self.cache_index["entries"][cache_key]
                    self._save_index()
                return None
                
            # Load directly to target device
            device = device or self.device
            tensors = torch.load(
                tensors_path,
                map_location=device
            )
            
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
                
            # Return with tensors already on correct device
            return {
                "pixel_values": tensors["pixel_values"],  # Already on device
                "prompt_embeds": tensors["prompt_embeds"],
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"],
                "original_size": metadata["original_size"],
                "crop_coords": metadata["crop_coords"],
                "target_size": metadata["target_size"],
                "text": metadata.get("text")
            }
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
            
    def cleanup(self, max_age: Optional[float] = None):
        """Clean up old cache entries.
        
        Args:
            max_age: Maximum age in seconds to keep entries
        """
        try:
            with self._lock:
                current_time = time.time()
                keys_to_remove = []
                
                for key, entry in self.cache_index["entries"].items():
                    if max_age and (current_time - entry["created_at"]) > max_age:
                        keys_to_remove.append(key)
                        
                    # Also check for missing files
                    latents_path = Path(entry["latents_path"])
                    metadata_path = Path(entry["metadata_path"]) 
                    if not latents_path.exists() or not metadata_path.exists():
                        keys_to_remove.append(key)
                        
                # Remove invalid entries
                for key in keys_to_remove:
                    entry = self.cache_index["entries"][key]
                    try:
                        Path(entry["latents_path"]).unlink(missing_ok=True)
                        Path(entry["metadata_path"]).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete cache files for {key}: {e}")
                    del self.cache_index["entries"][key]
                    
                self.cache_index["stats"]["last_cleanup"] = current_time
                self.cache_index["stats"]["total_entries"] = len(self.cache_index["entries"])
                self._save_index()
                
                logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

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
        """Batch save latents with optimized IO."""
        try:
            import concurrent.futures
            import asyncio
            import io
            import pickle
            
            results = []
            
            # Pre-allocate CPU tensors in one go
            cpu_tensors = [{
                k: v.cpu() for k, v in tensors.items()
            } for tensors in batch_tensors]
            
            # Optimize IO operations
            async def save_batch():
                async def save_single(tensors, path, metadata, idx):
                    try:
                        cache_key = self.get_cache_key(path)
                        tensors_path = self.latents_dir / f"{cache_key}.pt"
                        metadata_path = self.metadata_dir / f"{cache_key}.json"
                        
                        # Serialize tensor data to bytes buffer first
                        buffer = io.BytesIO()
                        torch.save(tensors, buffer)
                        tensor_bytes = buffer.getvalue()
                        
                        # Use ThreadPoolExecutor for parallel IO
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            # Write both files in parallel
                            await asyncio.gather(
                                asyncio.get_event_loop().run_in_executor(
                                    pool,
                                    lambda: tensors_path.write_bytes(tensor_bytes)
                                ),
                                asyncio.get_event_loop().run_in_executor(
                                    pool,
                                    lambda: json.dump(
                                        metadata,
                                        open(metadata_path, 'w'),
                                        indent=2
                                    )
                                )
                            )
                            
                            # Update index in memory only
                            self.cache_index["entries"][cache_key] = {
                                "tensors_path": str(tensors_path),
                                "metadata_path": str(metadata_path),
                                "created_at": time.time()
                            }
                            
                        return True
                        
                    except Exception as e:
                        logger.error(f"Failed to save cache entry: {e}")
                        return False
                
                # Process all items in parallel
                tasks = [
                    save_single(t, p, m, i) 
                    for i, (t, p, m) in enumerate(zip(cpu_tensors, original_paths, batch_metadata))
                ]
                return await asyncio.gather(*tasks)
            
            # Run batch save
            results = asyncio.run(save_batch())
            
            # Update index file periodically instead of every save
            if any(results):
                self._save_index()
                
            return results
            
        except Exception as e:
            logger.error(f"Batch save failed: {e}")
            return [False] * len(batch_tensors)
