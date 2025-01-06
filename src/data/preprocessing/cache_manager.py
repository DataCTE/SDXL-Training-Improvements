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
from src.data.preprocessing.exceptions import CacheError
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

    def save_tag_index(self, tag_data: Dict[str, Any]) -> None:
        """Save tag index data atomically with proper structure.
        
        Args:
            tag_data: Dictionary containing tag metadata and statistics
        """
        with self._lock:
            try:
                tag_stats_path = self.get_tag_statistics_path()
                tag_images_path = self.get_image_tags_path()
                
                # Split data into appropriate files
                stats_data = {
                    "version": tag_data["version"],
                    "metadata": tag_data["metadata"],
                    "statistics": tag_data["statistics"]
                }
                
                images_data = {
                    "version": tag_data["version"],
                    "updated_at": tag_data["updated_at"],
                    "images": tag_data["images"]
                }
                
                # Use atomic writes for both files
                self._atomic_json_save(tag_stats_path, stats_data)
                self._atomic_json_save(tag_images_path, images_data)
                
                # Update cache index with tag metadata
                self.cache_index["tag_metadata"] = {
                    "statistics": tag_data["statistics"],
                    "metadata": tag_data["metadata"],
                    "last_updated": time.time()
                }
                self._save_index()
                
                logger.info(f"Tag index saved successfully with {len(tag_data['images'])} entries")
                
            except Exception as e:
                raise CacheError("Failed to save tag index", context={
                    'tag_stats_path': str(tag_stats_path),
                    'tag_images_path': str(tag_images_path),
                    'error': str(e)
                })

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
        """Get path to tag statistics file."""
        return self.tags_dir / "tag_statistics.json"

    def get_image_tags_path(self) -> Path:
        """Get path to image tags file."""
        return self.tags_dir / "image_tags.json"

    def _atomic_json_save(self, path: Path, data: Dict[str, Any]) -> None:
        """Save JSON data atomically using a temporary file.
        
        Args:
            path: Path to save the JSON file
            data: Dictionary data to save
        """
        temp_path = path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(path)  # Atomic replace
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()  # Clean up temp file
            raise CacheError(f"Failed to save JSON file: {path}", context={
                'path': str(path),
                'error': str(e)
            })

    def verify_and_rebuild_cache(self, image_paths: List[str], captions: List[str]) -> None:
        """Verify cache integrity and rebuild if necessary."""
        logger.info("Starting cache verification...")
        needs_rebuild = False
        
        # Step 1: Check if tag metadata structure exists
        tag_stats_path = self.get_tag_statistics_path()
        tag_images_path = self.get_image_tags_path()
        
        # Initialize empty structures if files don't exist
        if not (tag_stats_path.exists() and tag_images_path.exists()):
            logger.info("Initializing new tag metadata structure...")
            try:
                self.initialize_tag_metadata()
                needs_rebuild = True
            except Exception as e:
                raise CacheError("Failed to initialize tag metadata structure", context={
                    'tag_stats_path': str(tag_stats_path),
                    'tag_images_path': str(tag_images_path),
                    'error': str(e)
                })
        
        # Step 2: Verify existing structure
        if not needs_rebuild:
            try:
                with self._lock:
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
                        logger.warning("Existing tag metadata missing required fields, rebuilding...")
                        needs_rebuild = True
                    
                    # Check image coverage if structure is valid
                    if not needs_rebuild:
                        missing_tags = []
                        missing_captions = []
                        for path, caption in zip(image_paths, captions):
                            if str(path) not in images_data.get("images", {}):
                                missing_tags.append(str(path))
                            if not caption:
                                missing_captions.append(str(path))
                            if len(missing_tags) > 5:  # Limit reported missing tags
                                break
                        
                        if missing_tags or missing_captions:
                            logger.warning("Missing tag data or captions, rebuilding cache...")
                            needs_rebuild = True
                            
            except Exception as e:
                logger.warning(f"Failed to verify existing tag metadata: {e}")
                needs_rebuild = True
        
        # Step 3: Rebuild if needed
        if needs_rebuild:
            logger.info("Starting cache rebuild...")
            try:
                with self._lock:
                    self.rebuild_cache_index()
                    
                    # Verify rebuild success
                    if not self._verify_rebuild_success():
                        raise CacheError("Cache rebuild failed", context={
                            'cache_dir': str(self.cache_dir),
                            'total_entries': len(self.cache_index.get("entries", {}))
                        })
                    logger.info("Cache rebuild completed successfully")
                    
            except Exception as e:
                raise CacheError("Failed to rebuild cache", context={
                    'cache_dir': str(self.cache_dir),
                    'error': str(e)
                })

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

    def initialize_tag_metadata(self) -> Dict[str, Any]:
        """Initialize empty tag metadata structure."""
        empty_stats = {
            "version": "1.0",
            "metadata": {},
            "statistics": {
                "total_samples": 0,
                "tag_type_counts": {},
                "unique_tags": {}
            }
        }
        empty_images = {
            "version": "1.0",
            "updated_at": time.time(),
            "images": {}
        }
        
        with self._lock:
            try:
                tag_stats_path = self.get_tag_statistics_path()
                tag_images_path = self.get_image_tags_path()
                
                # Use atomic writes for both files
                self._atomic_json_save(tag_stats_path, empty_stats)
                self._atomic_json_save(tag_images_path, empty_images)
                
                return {
                    "statistics": empty_stats,
                    "images": empty_images
                }
            except Exception as e:
                raise CacheError("Failed to initialize tag metadata", context={
                    'tag_stats_path': str(tag_stats_path),
                    'tag_images_path': str(tag_images_path),
                    'error': str(e)
                })

    def _validate_tag_metadata(self, stats_data: Dict[str, Any], images_data: Dict[str, Any]) -> bool:
        """Validate tag metadata structure and content."""
        try:
            # Required fields for validation
            required_stats_fields = ["metadata", "statistics", "version"]
            required_images_fields = ["images", "version", "updated_at"]
            required_stats = ["total_samples", "tag_type_counts", "unique_tags"]
            
            # Validate basic structure
            if not (all(field in stats_data for field in required_stats_fields) and
                   all(field in images_data for field in required_images_fields)):
                return False
                
            # Validate statistics structure
            if not all(field in stats_data["statistics"] for field in required_stats):
                return False
                
            # Validate version compatibility
            if (stats_data.get("version") != "1.0" or 
                images_data.get("version") != "1.0"):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Tag metadata validation failed: {e}")
            return False

    def verify_tag_cache(self, image_paths: List[str], captions: List[str]) -> bool:
        """Verify tag cache integrity and coverage."""
        try:
            tag_stats_path = self.get_tag_statistics_path()
            tag_images_path = self.get_image_tags_path()
            
            if not (tag_stats_path.exists() and tag_images_path.exists()):
                return False
            
            # Load and verify tag data
            with self._lock:
                with open(tag_stats_path, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                with open(tag_images_path, 'r', encoding='utf-8') as f:
                    images_data = json.load(f)
                    
                # Verify structure and version
                if not self._validate_tag_metadata(stats_data, images_data):
                    return False
                    
                # Check coverage
                image_tags = images_data.get("images", {})
                return all(str(path) in image_tags for path in image_paths)
                
        except Exception as e:
            logger.error(f"Tag cache verification failed: {e}")
            return False
