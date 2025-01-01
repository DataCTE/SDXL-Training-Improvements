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
from PIL import Image
import numpy as np

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

    def _save_tensor_file(self, tensors: Dict[str, torch.Tensor], path: Path) -> bool:
        """Common tensor saving logic."""
        try:
            torch.save(tensors, path, weights_only=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensors: {e}")
            return False

    def save_vae_latents(self, tensors: Dict[str, torch.Tensor], original_path: Union[str, Path], metadata: Dict[str, Any]) -> bool:
        """Save VAE encoded latents to cache.
        
        Args:
            tensors: Dict containing 'vae_latents' key with VAE encoded tensor
            original_path: Original image path
            metadata: Additional metadata to store
        """
        cache_key = self.get_cache_key(original_path)
        tensors_path = self.latents_dir / f"{cache_key}.pt"
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        
        try:
            # Only save VAE encoded latents
            tensors_to_save = {
                "vae_latents": tensors["vae_latents"].cpu()  # Save VAE latents
            }
            
            if not self._save_tensor_file(tensors_to_save, tensors_path):
                return False
                
            full_metadata = {
                "original_path": str(original_path),
                "created_at": time.time(),
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f)
            
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "tensors_path": str(tensors_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time(),
                    "is_valid": True,
                    "last_checked": time.time()
                }
                self._update_stats()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save VAE latents to cache: {e}")
            return False

    def _load_tensor_file(self, path: Path, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Common tensor loading logic."""
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            logger.error(f"Failed to load tensors: {e}")
            return None

    def _validate_and_clean_cache_entry(self, tensor_path: Path, metadata_path: Path, cache_key: str) -> bool:
        """Validate cache files and clean up if invalid."""
        if not tensor_path.exists() or not metadata_path.exists():
            with self._lock:
                if cache_key in self.cache_index["entries"]:
                    try:
                        tensor_path.unlink(missing_ok=True)
                        metadata_path.unlink(missing_ok=True)
                        del self.cache_index["entries"][cache_key]
                        self._save_index()
                    except Exception as e:
                        logger.error(f"Failed to clean cache entry {cache_key}: {e}")
            return False
        return True

    def _get_cache_size(self) -> int:
        """Get total size of cached files."""
        return sum(os.path.getsize(str(p)) for p in self.latents_dir.glob("*.pt"))

    def load_image_to_tensor(self, image_path: Union[str, Path], device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load and convert image to tensor with proper device placement.
        
        Args:
            image_path: Path to image file
            device: Target device for tensor
            
        Returns:
            Tuple of (tensor, (width, height))
        """
        try:
            device = device or self.device
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            
            # Convert to tensor efficiently
            img_tensor = torch.from_numpy(np.array(img)).float().to(device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
            
            return img_tensor, (w, h)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def load_tensors(self, cache_key: str, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
        """Load cached tensors with validation.
        
        Args:
            cache_key: Cache key for the entry
            device: Target device for tensors
            
        Returns:
            Dict containing tensors and metadata or None if invalid
        """
        try:
            entry = self.cache_index["entries"].get(cache_key)
            if not entry or not entry.get("is_valid", False):
                return None

            tensor_path = Path(entry["tensors_path"])
            metadata_path = Path(entry["metadata_path"])
            
            if not self._validate_and_clean_cache_entry(tensor_path, metadata_path, cache_key):
                return None

            device = device or self.device
            tensors = self._load_tensor_file(tensor_path, device)
            if tensors is None:
                return None

            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                self._invalidate_cache_entry(cache_key)
                return None

            # Validate required keys
            required_keys = {
                "tensors": {"vae_latents"},
                "metadata": {"original_size", "crop_coords", "target_size"}
            }
            
            if not (all(k in tensors for k in required_keys["tensors"]) and 
                   all(k in metadata for k in required_keys["metadata"])):
                self._invalidate_cache_entry(cache_key)
                return None

            return {"metadata": metadata, **tensors}

        except Exception as e:
            logger.error(f"Unexpected error loading cache for {cache_key}: {str(e)}")
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
        """Update cache statistics."""
        cache_size = self._get_cache_size()
        stats = {
            "total_entries": len(self.cache_index["entries"]),
            "latents_size": cache_size,
            "last_updated": time.time()
        }
        self.cache_index["stats"].update(stats)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return {
            "total_entries": len(self.cache_index["entries"]),
            "cache_size_bytes": self._get_cache_size(),
            "last_updated": self.cache_index["last_updated"],
            "last_cleanup": self.cache_index["stats"]["last_cleanup"]
        }

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
