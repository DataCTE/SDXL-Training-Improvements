"""Cache management for preprocessed latents with WSL support."""
from pathlib import Path
import json
import time
import torch
import threading
from typing import Dict, Optional, Union, Any, List, Tuple
from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.data.config import Config
import hashlib
logger = get_logger(__name__)

class CacheManager:
    """Manages caching of preprocessed latents with WSL path support."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: Optional["Config"] = None,
        max_cache_size: int = 10000,
        device: Optional[torch.device] = None
    ):
        """Initialize cache manager."""
        self.cache_dir = Path(convert_windows_path(cache_dir))  # Convert path for WSL
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Core setup
        self._lock = threading.Lock()
        
        # Create subdirectories
        self.latents_dir = self.cache_dir / "latents"
        self.metadata_dir = self.cache_dir / "metadata"
        self.latents_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize cache state
        self.index_path = self.cache_dir / "cache_index.json"
        self.rebuild_cache_index()

    def __getstate__(self):
        """Customize pickling behavior."""
        state = self.__dict__.copy()
        # Don't pickle the lock
        if '_lock' in state:
            del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Customize unpickling behavior."""
        self.__dict__.update(state)
        # Recreate the lock in the new process
        self._lock = threading.Lock()

    def rebuild_cache_index(self) -> None:
        """Rebuild cache index from disk as source of truth."""
        new_index = {
            "version": "1.0",
            "created_at": time.time(),
            "last_updated": time.time(),
            "entries": {},
            "stats": {"total_entries": 0, "latents_size": 0}
        }
        
        # Scan latents directory
        for latents_path in self.latents_dir.glob("*.pt"):
            metadata_path = self.metadata_dir / f"{latents_path.stem}.json"
            
            if not (latents_path.exists() and metadata_path.exists()):
                continue
                
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                cache_key = self.get_cache_key(metadata["original_path"])
                new_index["entries"][cache_key] = {
                    "tensors_path": str(latents_path),
                    "metadata_path": str(metadata_path),
                    "created_at": metadata.get("created_at", time.time()),
                    "bucket_dims": metadata.get("bucket_dims"),
                    "is_valid": True
                }
            except Exception as e:
                logger.warning(f"Failed to process cache file {latents_path}: {e}")
        
        self.cache_index = new_index
        self._save_index()

    def get_uncached_paths(self, image_paths: List[str]) -> List[str]:
        """Get list of paths that need processing."""
        uncached = []
        for path in image_paths:
            cache_key = self.get_cache_key(path)
            cache_entry = self.cache_index["entries"].get(cache_key)
            
            if not cache_entry or not self._validate_cache_entry(cache_entry):
                uncached.append(path)
                
        return uncached

    def save_latents(
        self, 
        tensors: Dict[str, torch.Tensor],
        path: Union[str, Path],
        metadata: Dict[str, Any],
        bucket_dims: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Save processed tensors and metadata to cache."""
        cache_key = self.get_cache_key(path)
        tensors_path = self.latents_dir / f"{cache_key}.pt"
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        
        try:
            # Save tensors
            torch.save({k: v.cpu() for k, v in tensors.items()}, tensors_path)
            
            # Save metadata
            full_metadata = {
                "original_path": str(path),
                "created_at": time.time(),
                "bucket_dims": bucket_dims,
                **metadata
            }
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f)
            
            # Update index
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "tensors_path": str(tensors_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time(),
                    "bucket_dims": bucket_dims,
                    "is_valid": True
                }
                self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return False

    def load_tensors(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached tensors and metadata."""
        entry = self.cache_index["entries"].get(cache_key)
        if not entry or not self._validate_cache_entry(entry):
            return None
            
        try:
            # Load and create copies of tensors and metadata
            tensors = {k: v.clone() for k, v in torch.load(entry["tensors_path"], map_location=self.device).items()}
            with open(entry["metadata_path"]) as f:
                metadata = json.loads(f.read())  # Creates a new dict
            return {"metadata": metadata, **tensors}
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
            return None

    def _validate_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate cache entry files exist and are valid."""
        return (
            Path(entry["tensors_path"]).exists() and
            Path(entry["metadata_path"]).exists() and
            entry.get("is_valid", False)
        )

    def _save_index(self) -> None:
        """Save cache index to disk atomically."""
        temp_path = self.index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
            temp_path.replace(self.index_path)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def get_cache_key(self, path: Union[str, Path]) -> str:
        """Generate cache key from path."""
        path_str = str(path)
        return hashlib.md5(path_str.encode()).hexdigest()

    def save_tag_index(self, index_data: Dict[str, Any], index_path: Optional[Path] = None) -> bool:
        """Save tag index with split files for better performance."""
        try:
            # Split data into statistics and image tags
            statistics_data = {
                "metadata": index_data["metadata"],
                "statistics": index_data["statistics"]
            }
            image_tags_data = {
                "images": index_data["images"]
            }
            
            # Save statistics to cache/tags/statistics.json
            stats_path = self.get_tag_statistics_path()
            self._atomic_json_save(stats_path, statistics_data)
            
            # Save image tags to cache/tags/image_tags.json
            tags_path = self.get_image_tags_path()
            self._atomic_json_save(tags_path, image_tags_data)
            
            # Update cache index
            with self._lock:
                self.cache_index["tag_stats_path"] = str(stats_path)
                self.cache_index["image_tags_path"] = str(tags_path)
                self.cache_index["tag_index_updated"] = time.time()
                self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tag index: {e}")
            return False

    def _atomic_json_save(self, path: Path, data: Dict[str, Any]) -> None:
        """Save JSON data atomically with proper formatting."""
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def get_tag_index_path(self) -> Path:
        """Get path for tag index directory."""
        tags_dir = self.cache_dir / "tags"
        tags_dir.mkdir(exist_ok=True)
        return tags_dir

    def get_tag_statistics_path(self) -> Path:
        """Get path for tag statistics file."""
        return self.get_tag_index_path() / "statistics.json"

    def get_image_tags_path(self) -> Path:
        """Get path for image tags file."""
        return self.get_tag_index_path() / "image_tags.json"

    def load_tag_index(self) -> Optional[Dict[str, Any]]:
        """Load tag index from split files."""
        try:
            # Load statistics
            stats_path = self.get_tag_statistics_path()
            if not stats_path.exists():
                return None
            
            with open(stats_path, 'r', encoding='utf-8') as f:
                statistics_data = json.load(f)
            
            # Load image tags
            tags_path = self.get_image_tags_path()
            if tags_path.exists():
                with open(tags_path, 'r', encoding='utf-8') as f:
                    image_tags_data = json.load(f)
            else:
                image_tags_data = {"images": {}}
            
            # Combine data
            return {
                "metadata": statistics_data["metadata"],
                "statistics": statistics_data["statistics"],
                "images": image_tags_data["images"]
            }
            
        except Exception as e:
            logger.error(f"Failed to load tag index: {e}")
            return None

    def verify_and_rebuild_cache(self, image_paths: List[str], captions: List[str]) -> None:
        """Verify cache integrity and rebuild if needed."""
        logger.info("Verifying cache integrity...")
        
        # Create image-caption mapping
        image_captions = dict(zip(image_paths, captions))
        
        # Check uncached images
        uncached = self.get_uncached_paths(image_paths)
        missing_count = len(uncached)
        
        if missing_count > 0:
            logger.warning(
                f"Found {missing_count} uncached images "
                f"({missing_count/len(image_paths)*100:.1f}% of dataset)"
            )
        
        # Verify existing cache entries
        invalid_entries = []
        for path in image_paths:
            cache_key = self.get_cache_key(path)
            entry = self.cache_index["entries"].get(cache_key)
            
            if entry and not self._validate_cache_entry(entry):
                invalid_entries.append(path)
            
            # Update caption in metadata if needed
            if entry and entry.get("metadata_path"):
                try:
                    with open(entry["metadata_path"], 'r') as f:
                        metadata = json.load(f)
                    metadata["caption"] = image_captions[path]
                    with open(entry["metadata_path"], 'w') as f:
                        json.dump(metadata, f)
                except Exception as e:
                    logger.warning(f"Failed to update caption for {path}: {e}")
        
        if invalid_entries:
            logger.warning(f"Found {len(invalid_entries)} invalid cache entries")
        
        # Update cache index
        if uncached or invalid_entries:
            logger.info("Rebuilding cache index...")
            self.rebuild_cache_index()
        
        # Update statistics
        self.cache_index["stats"]["total_entries"] = len(self.cache_index["entries"])
        self.cache_index["last_updated"] = time.time()
        self._save_index()
        
        logger.info(
            f"Cache verification complete. "
            f"Valid entries: {len(self.cache_index['entries'])}"
        )
