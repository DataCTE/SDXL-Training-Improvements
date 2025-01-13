"""Cache management for preprocessed latents with WSL support."""
from pathlib import Path
import json
import time
import torch
import threading
from typing import Dict, Optional, Union, Any, List, Tuple
from src.core.logging import UnifiedLogger, LogConfig, ProgressPredictor, setup_logging
from src.data.utils.paths import convert_windows_path, convert_path_to_pathlib, convert_paths
from src.data.config import Config
import hashlib
from src.data.preprocessing.bucket_types import BucketDimensions, BucketInfo
from collections import defaultdict
from src.data.preprocessing.exceptions import CacheError
import os
import zlib

logger = setup_logging(
    LogConfig(
        name=__name__,
        enable_progress=True,
        enable_metrics=True,
        enable_memory=True
    )
)

def _get_json_encoder():
    """Get the best available JSON encoder."""
    try:
        import orjson
        return orjson
    except ImportError:
        return None

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
        self.cache_dir = convert_path_to_pathlib(cache_dir, make_absolute=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device not available. This model requires a GPU to run efficiently.")
            device = torch.device("cuda")
        self.device = device
        self.config = config
        
        # Try to get orjson for better performance
        self._json_encoder = _get_json_encoder()
        
        # Initialize lock first
        self._lock = threading.Lock()
        
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
        self.cache_index = self._load_cache_index()

    def __getstate__(self):
        """Customize pickling behavior."""
        state = self.__dict__.copy()
        # Don't pickle the lock
        if '_lock' in state:
            del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Customize unpickling behavior."""
        # Initialize lock first
        self._lock = threading.Lock()
        # Then update the rest of the state
        self.__dict__.update(state)

    def rebuild_cache_index(self) -> None:
        """Rebuild cache index from disk as source of truth."""
        logger.info("Starting rebuild_cache_index")
        
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
            "bucket_stats": {}
        }
        
        # First, verify and load tag metadata
        logger.info("Verifying tag metadata...")
        tag_stats_path = self.get_tag_statistics_path()
        tag_images_path = self.get_image_tags_path()
        
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
                    logger.info("Tag metadata verified successfully")
            except Exception as e:
                logger.warning(f"Tag metadata verification failed: {e}")
        else:
            logger.warning("Tag metadata invalid or missing, will need to be rebuilt")
            new_index["tag_metadata"] = {
                "statistics": {},
                "metadata": {},
                "images": {}
            }
        
        # Now scan VAE latents directory for primary files
        logger.info("Scanning latent cache...")
        vae_files = list(self.vae_latents_dir.glob("*.pt"))
        
        if not vae_files:
            logger.info("No cached files found, initializing empty cache")
            self.cache_index = new_index
            self._save_index()
            return
            
        # Process files with progress tracking
        total_files = len(vae_files)
        logger.info(f"Found {total_files} files to process")
        
        for i, vae_path in enumerate(vae_files, 1):
            try:
                cache_key = vae_path.stem
                clip_path = self.clip_latents_dir / f"{cache_key}.pt"
                metadata_path = self.metadata_dir / f"{cache_key}.json"
                
                if not (vae_path.exists() and clip_path.exists() and metadata_path.exists()):
                    continue
                
                # Get file sizes
                sizes = {
                    'vae': vae_path.stat().st_size,
                    'clip': clip_path.stat().st_size,
                    'metadata': metadata_path.stat().st_size
                }
                
                # Read metadata
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Update bucket statistics - ensure string keys
                if metadata.get("bucket_info"):
                    bucket_idx = metadata["bucket_info"].get("bucket_index")
                    if bucket_idx is not None:
                        bucket_key = str(bucket_idx)  # Convert to string
                        if bucket_key not in new_index["bucket_stats"]:
                            new_index["bucket_stats"][bucket_key] = 0
                        new_index["bucket_stats"][bucket_key] += 1
                
                # Create entry
                entry = {
                    "vae_latent_path": str(vae_path.relative_to(self.latents_dir)),
                    "clip_latent_path": str(clip_path.relative_to(self.latents_dir)),
                    "metadata_path": str(metadata_path.relative_to(self.latents_dir)),
                    "created_at": metadata.get("created_at", time.time()),
                    "is_valid": True,
                    "file_sizes": {
                        "vae": sizes['vae'],
                        "clip": sizes['clip'],
                        "metadata": sizes['metadata'],
                        "total": sum(sizes.values())
                    },
                    "bucket_info": metadata.get("bucket_info"),
                    "tag_reference": metadata.get("tag_reference")
                }
                
                # Update index
                new_index["entries"][cache_key] = entry
                new_index["stats"]["total_entries"] += 1
                new_index["stats"]["total_size"] += entry["file_sizes"]["total"]
                new_index["stats"]["latents_size"] += sizes['vae'] + sizes['clip']
                new_index["stats"]["metadata_size"] += sizes['metadata']
                
                # Log progress every 5%
                if i % max(total_files // 20, 1) == 0:
                    progress = (i / total_files) * 100
                    logger.info(f"Processing files: {i}/{total_files} ({progress:.1f}%)")
                
            except Exception as e:
                logger.warning(f"Failed to process cache files for {vae_path.stem}: {e}")
                continue
        
        # Save the new index
        logger.info("Saving cache index...")
        self.cache_index = new_index
        self._save_index()
        
        # Log final statistics
        logger.info(f"Cache rebuild completed with {new_index['stats']['total_entries']} entries")

    def get_uncached_paths(self, image_paths: List[str]) -> List[str]:
        """Get list of paths that need processing."""
        logger.info(f"Checking cache status for {len(image_paths)} paths...")
        start_time = time.time()
        
        uncached = []
        processed = 0
        last_log = time.time()
        log_interval = 5.0  # Log every 5 seconds
        
        for path in image_paths:
            path_str = str(convert_path_to_pathlib(path))
            cache_key = self.get_cache_key(path_str)
            cache_entry = self.cache_index["entries"].get(cache_key)
            
            if not cache_entry or not self._validate_cache_entry(cache_entry):
                uncached.append(path)
            
            processed += 1
            current_time = time.time()
            
            # Log progress periodically
            if current_time - last_log > log_interval:
                progress = (processed / len(image_paths)) * 100
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(image_paths) - processed) / rate if rate > 0 else 0
                
                logger.info(
                    f"Cache check progress: {processed}/{len(image_paths)} ({progress:.1f}%) "
                    f"Found {len(uncached)} uncached (Rate: {rate:.1f} files/s, ETA: {eta:.1f}s)"
                )
                last_log = current_time
        
        total_time = time.time() - start_time
        logger.info(
            f"Cache check completed in {total_time:.2f}s. "
            f"Found {len(uncached)} uncached paths out of {len(image_paths)} total"
        )
                
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
            logger.debug("Saving latents to cache", extra={
                'cache_key': cache_key,
                'path': str(path),
                'tensor_shapes': {k: v.shape for k,v in tensors.items()}
            })
            
            # Save VAE latents
            vae_path = convert_path_to_pathlib(self.vae_latents_dir / f"{cache_key}.pt")
            torch.save({
                "vae_latents": tensors["vae_latents"].cpu(),
                "time_ids": tensors["time_ids"].cpu()
            }, vae_path)
            
            logger.debug("Saved VAE latents", extra={
                'path': str(vae_path),
                'size': vae_path.stat().st_size
            })
            
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

    def load_tensors(self, cache_key: str) -> Dict[str, Any]:
        """Load cached tensors optimized for training."""
        entry = None
        try:
            # Get cache entry with proper locking
            with self._lock:
                entry = self.cache_index["entries"].get(cache_key)
                if not entry:
                    raise RuntimeError(f"Cache entry not found for key: {cache_key}")
            
            # Map path keys to entry keys
            path_mapping = {
                "vae": "vae_latent_path",
                "clip": "clip_latent_path",
                "metadata": "metadata_path"
            }
            
            # Convert and check paths
            paths = {}
            for key, entry_key in path_mapping.items():
                try:
                    relative_path = entry[entry_key]
                    full_path = self.latents_dir / relative_path
                    logger.debug(f"Processing path for {key}:\n  Relative: {relative_path}\n  Full: {full_path}")
                    
                    converted_path = convert_path_to_pathlib(full_path, make_absolute=True)
                    logger.debug(f"Converted path for {key}: {converted_path}")
                    
                    if not os.path.exists(converted_path):
                        raise RuntimeError(
                            f"File does not exist: {converted_path}\n"
                            f"Original relative path: {relative_path}\n"
                            f"Full path before conversion: {full_path}"
                        )
                    if os.path.getsize(converted_path) == 0:
                        raise RuntimeError(f"File is empty: {converted_path}")
                        
                    paths[key] = converted_path
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to process path for {key}:\n"
                        f"Original path: {entry[entry_key]}\n"
                        f"Latents dir: {self.latents_dir}\n"
                        f"Error: {str(e)}"
                    ) from e
            
            # Load files with detailed error reporting
            try:
                # Check VAE file
                vae_data = torch.load(paths["vae"], map_location=self.device)
                required_vae_keys = ["vae_latents", "time_ids"]
                if not all(k in vae_data for k in required_vae_keys):
                    raise RuntimeError(f"Invalid VAE data structure. Missing keys: {[k for k in required_vae_keys if k not in vae_data]}")
                
                # Check CLIP file    
                clip_data = torch.load(paths["clip"], map_location=self.device)
                required_clip_keys = ["prompt_embeds", "pooled_prompt_embeds"]
                if not all(k in clip_data for k in required_clip_keys):
                    raise RuntimeError(f"Invalid CLIP data structure. Missing keys: {[k for k in required_clip_keys if k not in clip_data]}")
                
                # Check metadata file
                with open(paths["metadata"], 'r', encoding='utf-8') as f:
                    metadata = json.loads(f.read())
                required_metadata_keys = ["text", "bucket_info"]
                if not all(k in metadata for k in required_metadata_keys):
                    raise RuntimeError(f"Invalid metadata structure. Missing keys: {[k for k in required_metadata_keys if k not in metadata]}")
                
                # Return the loaded and validated data
                return {
                    "vae_latents": vae_data["vae_latents"],
                    "prompt_embeds": clip_data["prompt_embeds"],
                    "pooled_prompt_embeds": clip_data["pooled_prompt_embeds"],
                    "time_ids": vae_data["time_ids"],
                    "metadata": {
                        "text": metadata.get("text"),
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
                raise RuntimeError(
                    f"Failed to load tensor files:\n"
                    f"VAE path: {paths.get('vae')}\n"
                    f"CLIP path: {paths.get('clip')}\n"
                    f"Metadata path: {paths.get('metadata')}\n"
                    f"Error: {str(e)}"
                ) from e
            
        except Exception as e:
            logger.error(
                f"Failed to load tensors for cache key {cache_key}:\n"
                f"Cache entry: {entry}\n"
                f"Latents dir: {self.latents_dir}\n"
                f"Error: {str(e)}",
                exc_info=True
            )
            raise

    def _validate_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate cache entry files exist and can be loaded."""
        try:
            # Validate entry structure
            required_fields = ["vae_latent_path", "clip_latent_path", "metadata_path", "is_valid"]
            if not all(field in entry for field in required_fields):
                logger.error(f"Missing required fields in cache entry: {[f for f in required_fields if f not in entry]}")
                return False
            
            if not entry["is_valid"]:
                logger.error("Cache entry marked as invalid")
                return False
            
            # Map path keys to entry keys
            path_mapping = {
                "vae": "vae_latent_path",
                "clip": "clip_latent_path",
                "metadata": "metadata_path"
            }
            
            # Convert and check paths
            paths = {}
            for key, entry_key in path_mapping.items():
                try:
                    relative_path = entry[entry_key]
                    full_path = self.latents_dir / relative_path
                    logger.debug(f"Processing path for {key}:\n  Relative: {relative_path}\n  Full: {full_path}")
                    
                    converted_path = convert_path_to_pathlib(full_path, make_absolute=True)
                    logger.debug(f"Converted path for {key}: {converted_path}")
                    
                    if not os.path.exists(converted_path):
                        logger.error(
                            f"Missing {key} file:\n"
                            f"Converted path: {converted_path}\n"
                            f"Original path: {relative_path}\n"
                            f"Full path: {full_path}"
                        )
                        return False
                        
                    if os.path.getsize(converted_path) == 0:
                        logger.error(f"Empty {key} file: {converted_path}")
                        return False
                        
                    paths[key] = converted_path
                    
                except Exception as e:
                    logger.error(
                        f"Failed to process path for {key}:\n"
                        f"Original path: {entry[entry_key]}\n"
                        f"Latents dir: {self.latents_dir}\n"
                        f"Error: {str(e)}"
                    )
                    return False
            
            # Try loading each file to verify contents
            try:
                # Check VAE file
                vae_data = torch.load(paths["vae"], map_location=self.device)
                required_vae_keys = ["vae_latents", "time_ids"]
                if not all(k in vae_data for k in required_vae_keys):
                    logger.error(f"Invalid VAE data structure. Missing keys: {[k for k in required_vae_keys if k not in vae_data]}")
                    return False
                
                # Check CLIP file    
                clip_data = torch.load(paths["clip"], map_location=self.device)
                required_clip_keys = ["prompt_embeds", "pooled_prompt_embeds"]
                if not all(k in clip_data for k in required_clip_keys):
                    logger.error(f"Invalid CLIP data structure. Missing keys: {[k for k in required_clip_keys if k not in clip_data]}")
                    return False
                
                # Check metadata file
                with open(paths["metadata"], 'r', encoding='utf-8') as f:
                    metadata = json.loads(f.read())
                required_metadata_keys = ["text", "bucket_info"]
                if not all(k in metadata for k in required_metadata_keys):
                    logger.error(f"Invalid metadata structure. Missing keys: {[k for k in required_metadata_keys if k not in metadata]}")
                    return False
                
            except Exception as e:
                logger.error(
                    f"Failed to load and validate file contents:\n"
                    f"VAE path: {paths.get('vae')}\n"
                    f"CLIP path: {paths.get('clip')}\n"
                    f"Metadata path: {paths.get('metadata')}\n"
                    f"Error: {str(e)}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error validating cache entry:\n"
                f"Error: {str(e)}\n"
                f"Entry: {entry}\n"
                f"Latents dir: {self.latents_dir}",
                exc_info=True
            )
            return False

    def _save_index(self) -> None:
        """Save cache index to disk with progress logging."""
        temp_path = self.index_path.with_suffix('.tmp')
        
        try:
            logger.info("Starting cache index save...")
            
            # JSON encoding
            logger.info("Encoding cache data to JSON...")
            if self._json_encoder:
                json_data = self._json_encoder.dumps(self.cache_index)
            else:
                json_data = json.dumps(
                    self.cache_index,
                    separators=(',', ':'),
                    ensure_ascii=False
                ).encode('utf-8')
            
            # Compress
            logger.info("Compressing data...")
            compressed_data = zlib.compress(json_data, level=1)
            
            # Write to temp file
            logger.info("Writing compressed data...")
            with open(temp_path, 'wb', buffering=1024*1024) as f:
                f.write(compressed_data)
            
            # Atomic replace
            logger.info("Performing atomic file replacement...")
            os.replace(str(temp_path), str(self.index_path))
            
            logger.info("Cache index saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
            
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _load_cache_index(self) -> Dict[str, Any]:
        """Load main cache index from disk with compression support."""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'rb') as f:
                    compressed_data = f.read()
                    
                # Decompress data
                json_data = zlib.decompress(compressed_data)
                
                # Use the best available JSON decoder
                if self._json_encoder:
                    return self._json_encoder.loads(json_data)
                else:
                    return json.loads(json_data)
        except zlib.error:
            # Handle old uncompressed format
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load uncompressed cache index: {e}")
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
            "bucket_stats": defaultdict(int),
            "tag_metadata": {
                "statistics": {},
                "metadata": {},
                "last_updated": time.time()
            }
        }

    def _log_cache_statistics(self, new_index: Dict[str, Any]) -> None:
        """Log cache statistics efficiently."""
        stats = new_index["stats"]
        
        # Prepare all statistics in memory first
        log_messages = [
            f"Cache index rebuilt with {stats['total_entries']} entries",
            f"Total cache size: {stats['total_size'] / (1024*1024):.2f} MB",
            f"Latents size: {stats['latents_size'] / (1024*1024):.2f} MB",
            f"Metadata size: {stats['metadata_size'] / (1024*1024):.2f} MB"
        ]
        
        # Add bucket distribution if exists
        if new_index["bucket_stats"]:
            log_messages.append("\nBucket distribution:")
            # Pre-sort and format bucket stats
            bucket_stats = sorted(new_index["bucket_stats"].items())
            log_messages.extend(
                f"Bucket {idx}: {count} images"
                for idx, count in bucket_stats
            )
        
        # Log all messages at once
        logger.info("\n".join(log_messages))

    def get_cache_key(self, path: Union[str, Path]) -> str:
        """Generate cache key from path."""
        path_str = str(convert_path_to_pathlib(path))
        return hashlib.md5(path_str.encode()).hexdigest()

    def save_tag_index(self, index_data: Dict[str, Any]) -> None:
        """Save tag index with validation."""
        try:
            logger.info("Saving tag index...")
            
            # Validate index data structure
            required_sections = ["metadata", "statistics"]
            for section in required_sections:
                if section not in index_data:
                    raise ValueError(f"Missing required section: {section}")
            
            # Ensure images section exists
            if "images" not in index_data:
                index_data["images"] = {}
            
            # Save to file
            tag_index_path = self.get_tag_statistics_path()
            tag_index_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tag_index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
                
            logger.info(f"Tag index saved to {tag_index_path}")
            
            # Save image tags separately for faster access
            if "images" in index_data:
                image_tags_path = self.get_image_tags_path()
                with open(image_tags_path, 'w') as f:
                    json.dump(index_data["images"], f, indent=2)
                logger.info(f"Image tags saved to {image_tags_path}")
                
        except Exception as e:
            logger.error(f"Failed to save tag index: {e}")
            raise

    def load_tag_index(self) -> Optional[Dict[str, Any]]:
        """Load tag index with validation."""
        try:
            logger.info("Loading tag index...")
            tag_index_path = self.get_tag_statistics_path()
            
            if not tag_index_path.exists():
                logger.warning(f"Tag index not found at {tag_index_path}")
                return None
            
            with open(tag_index_path) as f:
                index_data = json.load(f)
            
            # Validate structure
            required_sections = ["metadata", "statistics"]
            for section in required_sections:
                if section not in index_data:
                    logger.warning(f"Tag index missing required section: {section}")
                    return None
            
            # Load image tags if they exist
            image_tags_path = self.get_image_tags_path()
            if image_tags_path.exists():
                with open(image_tags_path) as f:
                    index_data["images"] = json.load(f)
                logger.info(f"Loaded {len(index_data['images'])} image tags")
            else:
                logger.warning("Image tags file not found")
                index_data["images"] = {}
            
            logger.info("Successfully loaded tag index")
            return index_data
            
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

    def verify_and_rebuild_cache(
        self,
        image_paths: List[Union[str, Path]],
        verify_existing: bool = True
    ) -> None:
        """Verify cache integrity and rebuild if necessary.
        
        Args:
            image_paths: List of image paths to verify
            verify_existing: If False, only verify entries that don't exist or are invalid
        """
        try:
            logger.info("Starting cache verification...")
            invalid_entries = []
            missing_entries = []
            
            # Check each image path
            for path in image_paths:
                cache_key = self.get_cache_key(path)
                entry = self.cache_index["entries"].get(cache_key)
                
                if not entry:
                    missing_entries.append(path)
                    continue
                    
                if not entry.get("is_valid", False):
                    invalid_entries.append(cache_key)
                    continue
                    
                # Skip verification of existing valid entries if not requested
                if not verify_existing:
                    continue
                    
                # Verify file existence and integrity
                try:
                    vae_path = self.latents_dir / entry["vae_latent_path"]
                    clip_path = self.latents_dir / entry["clip_latent_path"]
                    metadata_path = self.latents_dir / entry["metadata_path"]
                    
                    if not all(p.exists() for p in [vae_path, clip_path, metadata_path]):
                        invalid_entries.append(cache_key)
                        continue
                        
                    # Load files to verify integrity
                    try:
                        vae_data = torch.load(vae_path, map_location="cpu")
                        clip_data = torch.load(clip_path, map_location="cpu")
                        with open(metadata_path) as f:
                            metadata = json.loads(f.read())
                            
                        # Verify expected keys exist
                        if not all(k in vae_data for k in ["vae_latents", "time_ids"]):
                            invalid_entries.append(cache_key)
                            continue
                            
                        if not all(k in clip_data for k in ["prompt_embeds", "pooled_prompt_embeds"]):
                            invalid_entries.append(cache_key)
                            continue
                            
                        if not all(k in metadata for k in ["text", "bucket_info", "created_at"]):
                            invalid_entries.append(cache_key)
                            continue
                            
                    except Exception:
                        invalid_entries.append(cache_key)
                        continue
                        
                except Exception as e:
                    logger.warning(f"Failed to verify cache entry {cache_key}: {e}")
                    invalid_entries.append(cache_key)
            
            # Remove invalid entries
            if invalid_entries:
                logger.warning(f"Found {len(invalid_entries)} invalid cache entries")
                for key in invalid_entries:
                    entry = self.cache_index["entries"].get(key)
                    if entry:
                        # Remove associated files
                        try:
                            for path_key in ["vae_latent_path", "clip_latent_path", "metadata_path"]:
                                if path_key in entry:
                                    file_path = self.latents_dir / entry[path_key]
                                    if file_path.exists():
                                        file_path.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove files for {key}: {e}")
                        
                        # Remove from index
                        self.cache_index["entries"].pop(key, None)
                
                self._save_index()
            
            if missing_entries:
                logger.info(f"Found {len(missing_entries)} uncached entries")
            
            if not invalid_entries and not missing_entries:
                logger.info("Cache verification complete - all entries valid")
                
        except Exception as e:
            logger.error(f"Cache verification failed: {e}")
            raise CacheError("Failed to verify cache", context={"error": str(e)})

    def _verify_rebuild_success(self) -> Tuple[bool, Optional[str]]:
        """Verify that cache rebuild was successful."""
        try:
            # Verify basic cache structure
            if not isinstance(self.cache_index, dict):
                logger.error("Cache index is not a dictionary")
                return False, "Cache index is not a dictionary"
                
            required_fields = ["entries", "stats", "tag_metadata"]
            for field in required_fields:
                if field not in self.cache_index:
                    logger.error(f"Missing required field in cache index: {field}")
                    return False, f"Missing required field in cache index: {field}"
            
            # Initialize empty structures if needed
            if not self.cache_index["entries"]:
                self.cache_index["entries"] = {}
                
            if not self.cache_index["stats"]:
                self.cache_index["stats"] = {
                    "total_entries": 0,
                    "total_size": 0,
                    "latents_size": 0,
                    "metadata_size": 0
                }
                
            if not self.cache_index["tag_metadata"]:
                self.cache_index["tag_metadata"] = {
                    "statistics": {},
                    "metadata": {},
                    "last_updated": time.time()
                }
            
            # Ensure tag metadata files exist
            tag_stats_path = self.get_tag_statistics_path()
            tag_images_path = self.get_image_tags_path()
            
            if not tag_stats_path.exists():
                logger.info("Creating empty tag statistics file")
                self._atomic_json_save(tag_stats_path, {
                    "version": "1.0",
                    "metadata": {},
                    "statistics": {}
                })
                
            if not tag_images_path.exists():
                logger.info("Creating empty image tags file")
                self._atomic_json_save(tag_images_path, {
                    "version": "1.0",
                    "updated_at": time.time(),
                    "images": {}
                })
            
            # Save updated index
            self._save_index()
            
            logger.info("Cache validation successful", extra={
                'total_entries': self.cache_index["stats"]["total_entries"],
                'has_tag_metadata': bool(self.cache_index["tag_metadata"]),
                'tag_files_exist': tag_stats_path.exists() and tag_images_path.exists()
            })
            
            return True, None
            
        except Exception as e:
            error_msg = f"Failed to verify rebuild success: {e}"
            logger.error(error_msg)
            return False, error_msg

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
                
                # Update cache index with tag metadata
                self.cache_index["tag_metadata"] = {
                    "statistics": empty_stats["statistics"],
                    "metadata": empty_stats["metadata"],
                    "last_updated": time.time()
                }
                self._save_index()
                
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
        """Verify tag cache integrity and coverage with enhanced validation."""
        try:
            tag_stats_path = self.get_tag_statistics_path()
            tag_images_path = self.get_image_tags_path()
            
            if not (tag_stats_path.exists() and tag_images_path.exists()):
                logger.debug("Tag cache files not found")
                return False
            
            with self._lock:
                # Load and verify tag data
                try:
                    with open(tag_stats_path, 'r', encoding='utf-8') as f:
                        stats_data = json.load(f)
                    with open(tag_images_path, 'r', encoding='utf-8') as f:
                        images_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise CacheError("Invalid tag cache JSON format", context={
                        "error": str(e),
                        "stats_path": str(tag_stats_path),
                        "images_path": str(tag_images_path)
                    })
                    
                # Verify structure and version
                if not self._validate_tag_metadata(stats_data, images_data):
                    logger.warning("Tag metadata validation failed")
                    return False
                    
                # Check coverage with detailed logging
                image_tags = images_data.get("images", {})
                missing_paths = []
                missing_captions = []
                
                for path, caption in zip(image_paths, captions):
                    path_str = str(path)
                    if path_str not in image_tags:
                        missing_paths.append(path_str)
                    elif not caption:
                        missing_captions.append(path_str)
                
                if missing_paths or missing_captions:
                    if missing_paths:
                        logger.warning(f"Missing tags for {len(missing_paths)} images")
                        if len(missing_paths) <= 5:
                            logger.debug(f"Missing paths: {missing_paths}")
                    if missing_captions:
                        logger.warning(f"Missing captions for {len(missing_captions)} images")
                        if len(missing_captions) <= 5:
                            logger.debug(f"Missing captions: {missing_captions}")
                    return False
                    
                return True
                
        except Exception as e:
            logger.error(f"Tag cache verification failed: {e}", exc_info=True)
            return False

    def load_cache_index(self) -> Dict[str, Any]:
        """Public method to load cache index."""
        return self._load_cache_index()

    def is_cached(self, image_path: Union[str, Path]) -> bool:
        """Check if an image is properly cached with valid latents.
        
        Args:
            image_path: Path to the image to check
            
        Returns:
            bool: True if the image is cached and valid, False otherwise
        """
        try:
            cache_key = self.get_cache_key(image_path)
            entry = self.cache_index["entries"].get(cache_key)
            
            if not entry or not entry.get("is_valid", False):
                return False
                
            # Verify file existence
            vae_path = self.latents_dir / entry["vae_latent_path"]
            clip_path = self.latents_dir / entry["clip_latent_path"]
            metadata_path = self.latents_dir / entry["metadata_path"]
            
            return all(p.exists() for p in [vae_path, clip_path, metadata_path])
            
        except Exception:
            return False
