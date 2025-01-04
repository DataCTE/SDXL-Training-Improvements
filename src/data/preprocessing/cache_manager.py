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
from src.data.preprocessing.bucket_types import BucketDimensions, BucketInfo
logger = get_logger(__name__)

class CacheManager:
    """Manages caching of preprocessed latents with comprehensive bucket information."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: Optional["Config"] = None,
        max_cache_size: int = 10000,
        device: Optional[torch.device] = None
    ):
        """Initialize cache manager with bucket support."""
        self.cache_dir = Path(convert_windows_path(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Create separate directories for different types of latents
        self.vae_latents_dir = self.cache_dir / "vae"
        self.clip_latents_dir = self.cache_dir / "clip"
        self.metadata_dir = self.cache_dir / "metadata"
        self.bucket_info_dir = self.cache_dir / "buckets"
        self.tag_dir = self.cache_dir / "tags"
        
        # Create all required directories
        for directory in [self.vae_latents_dir, self.clip_latents_dir, 
                         self.metadata_dir, self.bucket_info_dir, self.tag_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
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
                entry = {
                    "tensors_path": str(latents_path),
                    "metadata_path": str(metadata_path),
                    "created_at": metadata.get("created_at", time.time()),
                    "is_valid": True
                }
                
                # Preserve bucket info if it exists in metadata
                if "bucket_info" in metadata:
                    entry["bucket_info"] = metadata["bucket_info"]
                
                new_index["entries"][cache_key] = entry
                
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
        bucket_info: Optional[BucketInfo] = None
    ) -> bool:
        """Save processed tensors with proper organization for training."""
        cache_key = self.get_cache_key(path)
        
        try:
            # Separate VAE and CLIP latents
            vae_tensors = {
                "vae_latents": tensors["vae_latents"].cpu(),  # [C, H, W]
                "time_ids": tensors["time_ids"].cpu(),        # [1, 6]
                "original_size": metadata.get("original_size"),
                "crop_coords": metadata.get("crop_coords", (0, 0)),
                "target_size": metadata.get("target_size")
            }
            
            clip_tensors = {
                "prompt_embeds": tensors["prompt_embeds"].cpu(),           # [77, 2048]
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"].cpu()  # [1, 1280]
            }
            
            # Save VAE latents
            vae_path = self.vae_latents_dir / f"{cache_key}.pt"
            torch.save(vae_tensors, vae_path)
            
            # Save CLIP latents
            clip_path = self.clip_latents_dir / f"{cache_key}.pt"
            torch.save(clip_tensors, clip_path)
            
            # Save metadata with paths and associations
            full_metadata = {
                "original_path": str(path),
                "created_at": time.time(),
                "vae_path": str(vae_path),
                "clip_path": str(clip_path),
                **metadata
            }
            
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            self._atomic_json_save(metadata_path, full_metadata)
            
            # Update cache index with bucket info
            with self._lock:
                self.cache_index["entries"][cache_key] = {
                    "vae_path": str(vae_path),
                    "clip_path": str(clip_path),
                    "metadata_path": str(metadata_path),
                    "created_at": time.time(),
                    "is_valid": True
                }
                
                if bucket_info:
                    self.cache_index["entries"][cache_key]["bucket_info"] = {
                        "dimensions": bucket_info.dimensions.__dict__,
                        "pixel_dims": bucket_info.pixel_dims,
                        "latent_dims": bucket_info.latent_dims,
                        "bucket_index": bucket_info.bucket_index,
                        "size_class": bucket_info.size_class,
                        "aspect_class": bucket_info.aspect_class
                    }
                
                self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return False

    def load_tensors(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached tensors optimized for DDPM and Flow Matching training."""
        entry = self.cache_index["entries"].get(cache_key)
        if not entry or not self._validate_cache_entry(entry):
            return None
        
        try:
            # Load VAE latents with proper shapes
            vae_data = torch.load(entry["vae_path"], map_location=self.device)
            vae_latents = vae_data["vae_latents"]  # [C, H, W]
            time_ids = vae_data["time_ids"]        # [1, 6]
            
            # Load CLIP embeddings with correct dimensions
            clip_data = torch.load(entry["clip_path"], map_location=self.device)
            prompt_embeds = clip_data["prompt_embeds"]           # [77, 2048]
            pooled_prompt_embeds = clip_data["pooled_prompt_embeds"]  # [1, 1280]
            
            # Load metadata
            with open(entry["metadata_path"]) as f:
                metadata = json.loads(f.read())
            
            # Return format compatible with both trainers
            return {
                # Core tensors
                "vae_latents": vae_latents,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "time_ids": time_ids,
                
                # Size information for SDXL conditioning
                "original_size": metadata.get("original_size"),
                "crop_coords": metadata.get("crop_coords", (0, 0)),
                "target_size": metadata.get("target_size"),
                
                # Additional info
                "metadata": metadata,
                "bucket_info": entry.get("bucket_info")
            }
            
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
            return None

    def _validate_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate cache entry files exist and are valid."""
        return (
            Path(entry["vae_path"]).exists() and
            Path(entry["clip_path"]).exists() and
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

    def verify_and_rebuild_cache(self, image_paths: List[str], captions: List[str]) -> None:
        """Verify cache integrity and rebuild if necessary."""
        logger.info("Verifying cache integrity...")
        needs_rebuild = False
        
        for path in tqdm(image_paths, desc="Verifying cache entries"):
            cache_key = self.get_cache_key(path)
            entry = self.cache_index["entries"].get(cache_key)
            
            if entry:
                # Check all required files exist
                if not (Path(entry["tensors_path"]).exists() and 
                       Path(entry["metadata_path"]).exists()):
                    needs_rebuild = True
                    break
                    
                # Verify bucket info structure if present
                if "bucket_info" in entry:
                    required_fields = {"dimensions", "pixel_dims", "latent_dims", 
                                     "bucket_index", "size_class", "aspect_class"}
                    if not all(field in entry["bucket_info"] for field in required_fields):
                        needs_rebuild = True
                        break
        
        if needs_rebuild:
            logger.warning("Cache verification failed. Rebuilding cache index...")
            self.rebuild_cache_index()
        else:
            logger.info("Cache verification completed successfully")

    def load_bucket_info(self, cache_key: str) -> Optional["BucketInfo"]:
        """Load cached bucket information."""
        entry = self.cache_index["entries"].get(cache_key)
        if not entry or not entry.get("bucket_info_path"):
            return None
            
        try:
            bucket_info_path = Path(entry["bucket_info_path"])
            if not bucket_info_path.exists():
                return None
                
            with open(bucket_info_path, 'r') as f:
                bucket_data = json.load(f)
            
            # Reconstruct BucketDimensions
            dimensions = BucketDimensions(
                width=bucket_data["dimensions"]["width"],
                height=bucket_data["dimensions"]["height"],
                width_latent=bucket_data["dimensions"]["width_latent"],
                height_latent=bucket_data["dimensions"]["height_latent"],
                aspect_ratio=bucket_data["dimensions"]["aspect_ratio"],
                aspect_ratio_inverse=bucket_data["dimensions"]["aspect_ratio_inverse"],
                total_pixels=bucket_data["dimensions"]["total_pixels"],
                total_latents=bucket_data["dimensions"]["total_latents"]
            )
            
            # Reconstruct BucketInfo
            return BucketInfo(
                dimensions=dimensions,
                pixel_dims=tuple(bucket_data["pixel_dims"]),
                latent_dims=tuple(bucket_data["latent_dims"]),
                bucket_index=bucket_data["bucket_index"],
                size_class=bucket_data["size_class"],
                aspect_class=bucket_data["aspect_class"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load bucket info: {e}")
            return None
