"""Cache management for preprocessed latents with WSL support."""
from pathlib import Path
import json
import time
import torch
import threading
from typing import Dict, Optional, Union, Any, List
from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path

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
            max_cache_size: Maximum number of items to keep in cache
            device: Torch device for tensor operations
        """
        # Convert cache directory path if needed
        self.cache_dir = convert_windows_path(cache_dir) if is_windows_path(cache_dir) else Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache subdirectories
        self.latents_dir = self.cache_dir / "latents"
        self.latents_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.cache_dir / "metadata" 
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Cache settings
        self.max_cache_size = max_cache_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize cache state
        self._initialize_cache()
        
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
        
    def _save_index(self):
        """Save cache index to disk with thread safety."""
        with self._lock:
            try:
                self.cache_index["last_updated"] = time.time()
                with open(self.index_path, 'w') as f:
                    json.dump(self.cache_index, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save cache index: {e}")

    def get_cache_key(self, path: Union[str, Path]) -> str:
        """Generate cache key from path."""
        path = convert_windows_path(path) if is_windows_path(path) else Path(path)
        return path.stem

    def save_latents(
        self,
        tensors: Dict[str, torch.Tensor],
        original_path: Union[str, Path],
        metadata: Dict[str, Any]
    ) -> bool:
        """Save tensors and metadata to cache.
        
        Args:
            tensors: Dictionary of tensors to cache
            original_path: Original file path (used for cache key)
            metadata: Additional metadata to store
            
        Returns:
            bool: Success status
        """
        try:
            cache_key = self.get_cache_key(original_path)
            
            # Save tensors
            tensors_path = self.latents_dir / f"{cache_key}.pt"
            torch.save(tensors, tensors_path)
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            full_metadata = {
                "original_path": str(original_path),
                "created_at": time.time(),
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update index
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "tensors_path": str(tensors_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time()
                }
                self.cache_index["stats"]["total_entries"] = len(self.cache_index["entries"])
                self._save_index()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
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
            Dict containing tensors and metadata if found, None if not in cache
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
                
            # Load tensors
            tensors = torch.load(
                tensors_path,
                map_location=device or self.device
            )
            
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
                
            # Combine tensors and metadata
            return {**tensors, **metadata}
            
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
