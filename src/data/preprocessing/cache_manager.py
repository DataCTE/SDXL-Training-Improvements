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
from collections import defaultdict
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
        
        # Create proper subfolder structure
        self.tags_dir = self.cache_dir / "tags"
        self.latents_dir = self.cache_dir / "latents"
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create all subdirectories within latents
        self.vae_latents_dir = self.latents_dir / "vae"
        self.clip_latents_dir = self.latents_dir / "clip"
        self.metadata_dir = self.latents_dir / "metadata"
        self.bucket_info_dir = self.latents_dir / "buckets"
        
        # Create all required subdirectories
        for directory in [
            self.vae_latents_dir,
            self.clip_latents_dir,
            self.metadata_dir,
            self.tags_dir,
            self.bucket_info_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache index
        self.index_path = self.cache_dir / "cache_index.json"
        self._lock = threading.Lock()
        self.cache_index = self.load_cache_index()

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
            "stats": {
                "total_entries": 0,
                "total_size": 0,
                "latents_size": 0,
                "metadata_size": 0
            },
            "tag_metadata": {},
            "bucket_stats": defaultdict(int)
        }
        
        # First, verify and load tag metadata
        logger.info("Verifying tag metadata...")
        tag_stats_path = self.get_tag_statistics_path()
        tag_images_path = self.get_image_tags_path()
        
        tag_metadata_valid = False
        if tag_stats_path.exists() and tag_images_path.exists():
            try:
                with open(tag_stats_path, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                with open(tag_images_path, 'r', encoding='utf-8') as f:
                    images_data = json.load(f)
                    
                if (all(key in stats_data for key in ["metadata", "statistics"]) and
                    "images" in images_data):
                    new_index["tag_metadata"] = {
                        "statistics": stats_data["statistics"],
                        "metadata": stats_data["metadata"],
                        "images": images_data["images"]
                    }
                    tag_metadata_valid = True
                    logger.info("Tag metadata verified successfully")
            except Exception as e:
                logger.warning(f"Tag metadata verification failed: {e}")
        
        if not tag_metadata_valid:
            logger.warning("Tag metadata invalid or missing, will need to be rebuilt")
            new_index["tag_metadata"] = {
                "statistics": {},
                "metadata": {},
                "images": {}
            }
        
        # Now scan VAE latents directory for primary files
        logger.info("Scanning latent cache...")
        for vae_path in self.vae_latents_dir.glob("*.pt"):
            cache_key = vae_path.stem
            clip_path = self.clip_latents_dir / f"{cache_key}.pt"
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            
            # Only process if we have all required files
            if not (vae_path.exists() and clip_path.exists() and metadata_path.exists()):
                continue
            
            try:
                # Get file sizes
                vae_size = vae_path.stat().st_size
                clip_size = clip_path.stat().st_size
                metadata_size = metadata_path.stat().st_size
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Verify metadata structure
                required_fields = {
                    "vae_latent_path", "clip_latent_path", "text",
                    "bucket_info", "created_at", "tag_reference"
                }
                
                if not all(field in metadata for field in required_fields):
                    logger.warning(f"Incomplete metadata for {cache_key}, skipping")
                    continue
                
                # Update bucket statistics
                if metadata.get("bucket_info"):
                    bucket_idx = metadata["bucket_info"].get("bucket_index")
                    if bucket_idx is not None:
                        new_index["bucket_stats"][bucket_idx] += 1
                
                entry = {
                    "vae_latent_path": str(vae_path.relative_to(self.latents_dir)),
                    "clip_latent_path": str(clip_path.relative_to(self.latents_dir)),
                    "metadata_path": str(metadata_path.relative_to(self.latents_dir)),
                    "created_at": metadata.get("created_at", time.time()),
                    "is_valid": True,
                    "file_sizes": {
                        "vae": vae_size,
                        "clip": clip_size,
                        "metadata": metadata_size,
                        "total": vae_size + clip_size + metadata_size
                    },
                    "bucket_info": metadata.get("bucket_info"),
                    "tag_reference": metadata.get("tag_reference")
                }
                
                new_index["entries"][cache_key] = entry
                new_index["stats"]["total_entries"] += 1
                new_index["stats"]["total_size"] += entry["file_sizes"]["total"]
                new_index["stats"]["latents_size"] += vae_size + clip_size
                new_index["stats"]["metadata_size"] += metadata_size
                
            except Exception as e:
                logger.warning(f"Failed to process cache files for {cache_key}: {e}")
        
        # Update cache index
        with self._lock:
            self.cache_index = new_index
            self._save_index()
        
        # Log cache statistics
        logger.info(f"Cache index rebuilt with {new_index['stats']['total_entries']} entries")
        logger.info(f"Total cache size: {new_index['stats']['total_size'] / (1024*1024):.2f} MB")
        logger.info(f"Latents size: {new_index['stats']['latents_size'] / (1024*1024):.2f} MB")
        logger.info(f"Metadata size: {new_index['stats']['metadata_size'] / (1024*1024):.2f} MB")
        
        # Log bucket distribution
        if new_index["bucket_stats"]:
            logger.info("\nBucket distribution:")
            for bucket_idx, count in sorted(new_index["bucket_stats"].items()):
                logger.info(f"Bucket {bucket_idx}: {count} images")

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
        bucket_info: Optional[BucketInfo] = None,
        tag_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save processed tensors with proper organization for training."""
        try:
            cache_key = self.get_cache_key(path)
            
            # Save VAE latents in vae subfolder
            vae_path = self.vae_latents_dir / f"{cache_key}.pt"
            torch.save({
                "vae_latents": tensors["vae_latents"].cpu(),
                "time_ids": tensors["time_ids"].cpu()
            }, vae_path)
            
            # Save CLIP latents in clip subfolder
            clip_path = self.clip_latents_dir / f"{cache_key}.pt"
            torch.save({
                "prompt_embeds": tensors["prompt_embeds"].cpu(),
                "pooled_prompt_embeds": tensors["pooled_prompt_embeds"].cpu()
            }, clip_path)
            
            # Convert BucketInfo to serializable dict
            bucket_dict = None
            if bucket_info:
                bucket_dict = {
                    "dimensions": {
                        "width": bucket_info.dimensions.width,
                        "height": bucket_info.dimensions.height,
                        "width_latent": bucket_info.dimensions.width_latent,
                        "height_latent": bucket_info.dimensions.height_latent,
                        "aspect_ratio": bucket_info.dimensions.aspect_ratio,
                        "aspect_ratio_inverse": bucket_info.dimensions.aspect_ratio_inverse,
                        "total_pixels": bucket_info.dimensions.total_pixels,
                        "total_latents": bucket_info.dimensions.total_latents
                    },
                    "pixel_dims": list(bucket_info.pixel_dims),
                    "latent_dims": list(bucket_info.latent_dims),
                    "bucket_index": bucket_info.bucket_index,
                    "size_class": bucket_info.size_class,
                    "aspect_class": bucket_info.aspect_class
                }
            
            # Load existing tag index or create new
            tag_index = self.load_tag_index() or {
                "version": "1.0",
                "metadata": {},
                "images": {}
            }
            
            # Update metadata for this specific image
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            full_metadata = {
                "vae_latent_path": str(vae_path),
                "clip_latent_path": str(clip_path),
                "created_at": time.time(),
                "text": metadata.get("text"),
                "bucket_info": bucket_dict,
                "tag_reference": {
                    "cache_key": cache_key,
                    "has_tags": bool(tag_info)
                }
            }
            
            # Update tag index if we have tag information
            if tag_info:
                tag_index["images"][str(path)] = {
                    "cache_key": cache_key,
                    "tags": tag_info["tags"]
                }
                
                # Save updated tag index
                self._atomic_json_save(
                    self.tags_dir / "tag_index.json",
                    tag_index
                )
            
            # Save metadata
            self._atomic_json_save(metadata_path, full_metadata)
            
            # Update cache index with latent pair info
            with self._lock:
                entry = {
                    "vae_latent_path": str(vae_path.relative_to(self.latents_dir)),
                    "clip_latent_path": str(clip_path.relative_to(self.latents_dir)),
                    "metadata_path": str(metadata_path.relative_to(self.latents_dir)),
                    "created_at": time.time(),
                    "is_valid": True,
                    "bucket_info": bucket_dict,
                    "tag_info": tag_info
                }
                
                self.cache_index["entries"][cache_key] = entry
                self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return False

    def load_tensors(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached tensors optimized for training."""
        entry = self.cache_index["entries"].get(cache_key)
        if not entry or not self._validate_cache_entry(entry):
            return None
        
        try:
            # Load VAE latents and time_ids
            vae_path = self.latents_dir / entry["vae_latent_path"]
            vae_data = torch.load(vae_path, map_location=self.device)
            
            # Load CLIP embeddings
            clip_path = self.latents_dir / entry["clip_latent_path"]
            clip_data = torch.load(clip_path, map_location=self.device)
            
            # Load metadata
            metadata_path = self.latents_dir / entry["metadata_path"]
            with open(metadata_path) as f:
                metadata = json.loads(f.read())
            
            return {
                "vae_latents": vae_data["vae_latents"],
                "prompt_embeds": clip_data["prompt_embeds"],
                "pooled_prompt_embeds": clip_data["pooled_prompt_embeds"],
                "time_ids": vae_data["time_ids"],
                "metadata": {
                    "vae_latent_path": entry["vae_latent_path"],
                    "clip_latent_path": entry["clip_latent_path"],
                    "text": metadata["text"],
                    "bucket_info": entry["bucket_info"],
                    "tag_info": entry.get("tag_info", {
                        "tags": {
                            "subject": [],
                            "style": [],
                            "quality": [],
                            "technical": [],
                            "meta": []
                        }
                    })
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load cache entry: {e}")
            return None

    def _validate_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate cache entry files exist and are valid."""
        try:
            vae_path = self.latents_dir / entry["vae_latent_path"]
            clip_path = self.latents_dir / entry["clip_latent_path"]
            metadata_path = self.latents_dir / entry["metadata_path"]
            
            return (
                vae_path.exists() and
                clip_path.exists() and
                metadata_path.exists() and
                entry.get("is_valid", False)
            )
        except Exception:
            return False

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
        logger.info("Starting comprehensive cache verification...")
        needs_rebuild = False
        
        # Step 1: Verify tag metadata files and structure
        logger.info("Verifying tag metadata...")
        tag_stats_path = self.get_tag_statistics_path()
        tag_images_path = self.get_image_tags_path()
        
        if not (tag_stats_path.exists() and tag_images_path.exists()):
            logger.warning("Tag metadata files missing")
            needs_rebuild = True
        else:
            try:
                # Load and verify tag statistics
                with open(tag_stats_path, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                with open(tag_images_path, 'r', encoding='utf-8') as f:
                    images_data = json.load(f)
                
                # Verify required fields
                required_stats_fields = ["metadata", "statistics", "version"]
                required_images_fields = ["images", "version", "updated_at"]
                
                if not (all(field in stats_data for field in required_stats_fields) and
                       all(field in images_data for field in required_images_fields)):
                    logger.warning("Tag metadata files missing required fields")
                    needs_rebuild = True
                
                # Verify all images have tag entries
                if not needs_rebuild:
                    for path in image_paths:
                        if str(path) not in images_data.get("images", {}):
                            logger.warning(f"Missing tag data for: {path}")
                            needs_rebuild = True
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to verify tag metadata: {e}")
                needs_rebuild = True
        
        # Step 2: Verify latent cache entries and their integrity
        if not needs_rebuild:
            logger.info("Verifying latent cache entries...")
            for path in tqdm(image_paths, desc="Verifying cache entries"):
                cache_key = self.get_cache_key(path)
                entry = self.cache_index["entries"].get(cache_key)
                
                if not entry:
                    logger.warning(f"Missing cache entry for: {path}")
                    needs_rebuild = True
                    break
                    
                # Verify entry structure and files
                if not self._validate_cache_entry(entry):
                    logger.warning(f"Invalid cache entry for: {path}")
                    needs_rebuild = True
                    break
                    
                # Verify tag references
                if not entry.get("tag_reference"):
                    logger.warning(f"Missing tag reference for: {path}")
                    needs_rebuild = True
                    break
        
        # Step 3: Verify cache index statistics
        if not needs_rebuild:
            logger.info("Verifying cache index statistics...")
            try:
                actual_entries = len(self.cache_index["entries"])
                reported_entries = self.cache_index["stats"]["total_entries"]
                
                if actual_entries != reported_entries:
                    logger.warning("Cache index statistics mismatch")
                    needs_rebuild = True
                    
            except Exception as e:
                logger.warning(f"Failed to verify cache statistics: {e}")
                needs_rebuild = True
        
        # Step 4: Rebuild if necessary
        if needs_rebuild:
            logger.warning("Cache verification failed. Starting complete rebuild...")
            self.rebuild_cache_index()
            
            # Verify rebuild was successful
            if not self._verify_rebuild_success():
                logger.error("Cache rebuild failed. Manual intervention may be required.")
            else:
                logger.info("Cache rebuild completed successfully")
        else:
            logger.info("Cache verification completed successfully - all systems valid")
        
    def _verify_rebuild_success(self) -> bool:
        """Verify that cache rebuild was successful."""
        try:
            # Check basic cache structure
            if not (self.cache_index and 
                    "entries" in self.cache_index and 
                    "stats" in self.cache_index and
                    "tag_metadata" in self.cache_index):
                return False
            
            # Verify tag metadata files exist
            if not (self.get_tag_statistics_path().exists() and 
                    self.get_image_tags_path().exists()):
                return False
            
            # Verify cache statistics make sense
            if self.cache_index["stats"]["total_entries"] <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify rebuild success: {e}")
            return False

    def load_bucket_info(self, cache_key: str) -> Optional["BucketInfo"]:
        """Load cached bucket information."""
        entry = self.cache_index["entries"].get(cache_key)
        if not entry or not entry.get("bucket_info"):
            return None
            
        try:
            bucket_info = entry["bucket_info"]
            
            # Reconstruct BucketDimensions from stored info
            dimensions = BucketDimensions(
                width=bucket_info["dimensions"]["width"],
                height=bucket_info["dimensions"]["height"],
                width_latent=bucket_info["dimensions"]["width_latent"],
                height_latent=bucket_info["dimensions"]["height_latent"],
                aspect_ratio=bucket_info["dimensions"]["aspect_ratio"],
                aspect_ratio_inverse=bucket_info["dimensions"]["aspect_ratio_inverse"],
                total_pixels=bucket_info["dimensions"]["total_pixels"],
                total_latents=bucket_info["dimensions"]["total_latents"]
            )
            
            # Reconstruct BucketInfo
            return BucketInfo(
                dimensions=dimensions,
                pixel_dims=tuple(bucket_info["pixel_dims"]),
                latent_dims=tuple(bucket_info["latent_dims"]),
                bucket_index=bucket_info["bucket_index"],
                size_class=bucket_info["size_class"],
                aspect_class=bucket_info["aspect_class"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load bucket info: {e}")
            return None

    def load_cache_index(self) -> Dict[str, Any]:
        """Load main cache index from disk or create new if not exists."""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
        
        # Return new cache index if loading fails or file doesn't exist
        return {
            "version": "1.0",
            "created_at": time.time(),
            "last_updated": time.time(),
            "entries": {},
            "stats": {
                "total_entries": 0,
                "total_size": 0,
                "latents_size": 0,
                "metadata_size": 0
            },
            "bucket_stats": defaultdict(int)
        }
