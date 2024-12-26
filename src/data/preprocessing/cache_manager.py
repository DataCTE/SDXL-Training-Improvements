"""High-performance cache management with extreme speedups."""
from pathlib import Path
import json
import time
import torch
import logging
from typing import Dict, Optional, Union, Any, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class TensorValidator:
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"Tensor {name} is None")
            
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0)
            logger.warning(f"Fixed NaN values in {name}")
            
        if torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
            logger.warning(f"Fixed Inf values in {name}")
            
        return tensor.contiguous()

@dataclass
class CacheStats:
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class CacheManager(TensorValidator):
    def __init__(self, cache_dir: Union[str, Path], **kwargs):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_latents_dir = self.cache_dir / "image_latents"
        self.text_latents_dir = self.cache_dir / "text_latents"
        self.image_latents_dir.mkdir(exist_ok=True)
        self.text_latents_dir.mkdir(exist_ok=True)
        self.index_path = self.cache_dir / "cache_index.json"
        self.stats = CacheStats()
        self.device = kwargs.get('device') or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_index()
        
        # Verify cache structure
        if not self.verify_cache_structure():
            logger.warning("Cache structure verification failed - rebuilding index")
            self._initialize_empty_index()
        
    def validate_cache_index(self) -> Tuple[List[str], List[str]]:
        """Validate cache index and return missing/invalid entries."""
        missing_text = []
        missing_latents = []
        valid_files = {}
        
        def validate_file(path: Path) -> bool:
            try:
                if not path.exists():
                    return False
                data = torch.load(path, map_location='cpu')
                return isinstance(data, dict) and ("latent" in data or "embeddings" in data)
            except:
                return False

        with ThreadPoolExecutor() as executor:
            for file_path, info in self.cache_index["files"].items():
                # Extract or create base_name from file_path if not in info
                if "base_name" not in info:
                    info["base_name"] = Path(file_path).stem
                    
                base_name = info["base_name"]
                
                # Get paths from cache entry or construct default paths
                image_latent_path = Path(info.get("image_latent_path", self.image_latents_dir / f"{base_name}.pt"))
                text_latent_path = Path(info.get("text_latent_path", self.text_latents_dir / f"{base_name}.pt"))
                
                # Validate files in parallel
                latent_future = executor.submit(validate_file, image_latent_path)
                text_future = executor.submit(validate_file, text_latent_path)
                
                valid_latent = latent_future.result()
                valid_text = text_future.result()
                
                if not valid_latent:
                    missing_latents.append(file_path)
                if not valid_text:
                    missing_text.append(file_path)
                    
                # Only keep valid entries
                if valid_latent or valid_text:
                    valid_files[file_path] = {
                        "base_name": base_name,
                        "timestamp": info.get("timestamp", time.time())
                    }
                    if valid_latent:
                        valid_files[file_path]["image_latent_path"] = str(image_latent_path)
                    if valid_text:
                        valid_files[file_path]["text_latent_path"] = str(text_latent_path)

        # Update cache index with only valid entries
        self.cache_index["files"] = valid_files
        self._save_index()
        
        logger.info(f"Cache validation complete. Found {len(missing_latents)} missing image latents and {len(missing_text)} missing text latents")
        return missing_text, missing_latents

    def _load_index(self):
        """Load or initialize the cache index."""
        try:
            if self.index_path.exists():
                with open(self.index_path) as f:
                    self.cache_index = json.load(f)
                    
                # Check if index is empty or invalid
                if not self.cache_index.get("files"):
                    logger.info("Empty or invalid cache index found - rebuilding from scan")
                    self.scan_and_rebuild_index()
            else:
                # Initialize new index and scan directories
                logger.info("No cache index found - creating new one from directory scan") 
                self.scan_and_rebuild_index()
                
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            self._initialize_empty_index()

    def _initialize_empty_index(self):
        """Initialize a new empty cache index with proper structure."""
        self.cache_index = {
            "metadata": {
                "created_at": time.time(),
                "last_updated": time.time(),
                "version": "1.0",
                "cache_dir": str(self.cache_dir),
                "image_latents_dir": str(self.image_latents_dir),
                "text_latents_dir": str(self.text_latents_dir)
            },
            "files": {},
            "statistics": {
                "total_files": 0,
                "total_image_latents": 0,
                "total_text_latents": 0
            }
        }
        self._save_index()

    def _save_index(self):
        """Save the cache index with updated metadata."""
        try:
            # Update metadata
            self.cache_index["metadata"]["last_updated"] = time.time()
            
            # Update statistics
            stats = self.cache_index["statistics"]
            stats["total_files"] = len(self.cache_index["files"])
            stats["total_image_latents"] = sum(
                1 for f in self.cache_index["files"].values() 
                if "image_latent_path" in f
            )
            stats["total_text_latents"] = sum(
                1 for f in self.cache_index["files"].values() 
                if "text_latent_path" in f
            )
            
            # Save with proper formatting
            with open(self.index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _process_latents(self, data: Union[Dict, torch.Tensor], prefix: str) -> Union[Dict, torch.Tensor]:
        if isinstance(data, dict):
            return {k: self.validate_tensor(v, f"{prefix}_{k}") for k, v in data.items() if isinstance(v, torch.Tensor)}
        elif isinstance(data, torch.Tensor):
            return self.validate_tensor(data, prefix)
        return data

    def save_preprocessed_data(
        self,
        image_latent: Optional[Dict[str, torch.Tensor]],
        text_latent: Optional[Dict[str, Any]] = None,
        metadata: Dict = None,
        file_path: Union[str, Path] = None
    ) -> bool:
        try:
            base_name = Path(file_path).stem
            metadata = metadata or {}
            metadata["timestamp"] = time.time()
            
            # Track files in cache index
            cache_entry = {
                "base_name": base_name,
                "timestamp": time.time()
            }

            if image_latent is not None:
                image_latent = self._process_latents(image_latent, "image")
                latent_path = self.image_latents_dir / f"{base_name}.pt"
                
                # Structure the latent data properly
                latent_data = {
                    "latent": {
                        "model_input": image_latent.get("image_latent", image_latent),
                        "latent": image_latent
                    },
                    "metadata": {
                        "original_size": metadata.get("original_size", (1024, 1024)),
                        "crop_top_left": metadata.get("crop_top_left", (0, 0)),
                        "target_size": metadata.get("target_size", (1024, 1024)),
                        **{k: v for k, v in metadata.items() if k not in ["original_size", "crop_top_left", "target_size"]}
                    }
                }
                
                torch.save(latent_data, latent_path)
                cache_entry["image_latent_path"] = str(latent_path)

            if text_latent is not None:
                text_latent = self._process_latents(text_latent.get("embeddings", {}), "text")
                text_path = self.text_latents_dir / f"{base_name}.pt"
                torch.save({
                    "embeddings": text_latent,
                    "metadata": {**metadata, "type": "text_latent"}
                }, text_path)
                cache_entry["text_latent_path"] = str(text_path)

            self.cache_index["files"][str(file_path)] = cache_entry
            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Failed to save data for {file_path}: {str(e)}")
            return False

    def get_cached_item(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            base_name = Path(file_path).stem
            result = {}

            def load_file(path: Path) -> Optional[Dict]:
                if not path.exists():
                    return None
                    
                data = torch.load(path, map_location='cpu')
                
                # Process latents and ensure format compatibility
                if "latent" in data:
                    latent_data = data["latent"]
                    if isinstance(latent_data, dict):
                        for k, v in latent_data.items():
                            if isinstance(v, torch.Tensor):
                                latent_data[k] = self._process_latents(v, f"cached_latent_{k}")
                    else:
                        data["latent"] = self._process_latents(latent_data, "cached_latent")
                    
                if "embeddings" in data:
                    data["embeddings"] = self._process_latents(data["embeddings"], "cached_emb")
                return data

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(load_file, self.image_latents_dir / f"{base_name}.pt"),
                    executor.submit(load_file, self.text_latents_dir / f"{base_name}.pt")
                ]

                for future in futures:
                    if data := future.result():
                        result.update(data)

            if not result:
                self.stats.cache_misses += 1
                return None

            if device is not None:
                result = self._to_device(result, device)

            self.stats.cache_hits += 1
            return result

        except Exception as e:
            logger.error(f"Failed to load cached item {file_path}: {str(e)}")
            self.stats.failed_items += 1
            return None

    def _to_device(self, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = self._to_device(v, device)
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            else:
                result[k] = v
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._save_index()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
    def scan_and_rebuild_index(self) -> None:
        """Scan cache directories and rebuild index with actual files."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Initialize new index structure
            new_index = {
                "metadata": {
                    "created_at": time.time(),
                    "last_updated": time.time(),
                    "version": "1.0",
                    "cache_dir": str(self.cache_dir),
                    "image_latents_dir": str(self.image_latents_dir),
                    "text_latents_dir": str(self.text_latents_dir)
                },
                "files": {},
                "statistics": {
                    "total_files": 0,
                    "total_image_latents": 0,
                    "total_text_latents": 0
                }
            }

            # Pre-scan directories to get file lists
            image_latents = list(self.image_latents_dir.glob("*.pt"))
            text_latents = list(self.text_latents_dir.glob("*.pt"))
            
            # Process files in parallel
            def process_latent_file(path: Path, is_image: bool) -> tuple[str, dict]:
                try:
                    # Just load metadata portion for speed
                    data = torch.load(path, map_location='cpu', weights_only=True)
                    if isinstance(data, dict):
                        base_name = path.stem
                        file_path = data.get("metadata", {}).get(
                            "path" if is_image else "image_path", 
                            str(path)
                        )
                        
                        entry = {
                            "base_name": base_name,
                            "timestamp": data.get("metadata", {}).get("timestamp", time.time())
                        }
                        
                        if is_image:
                            entry["image_latent_path"] = str(path)
                        else:
                            entry["text_latent_path"] = str(path)
                            
                        return file_path, entry
                        
                except Exception as e:
                    logger.warning(f"Failed to process {'image' if is_image else 'text'} latent {path}: {e}")
                return None

            # Process files in parallel using thread pool
            with ThreadPoolExecutor(max_workers=min(32, (len(image_latents) + len(text_latents)))) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(process_latent_file, path, True): path 
                    for path in image_latents
                }
                future_to_path.update({
                    executor.submit(process_latent_file, path, False): path 
                    for path in text_latents
                })

                # Process results as they complete
                for future in as_completed(future_to_path):
                    result = future.result()
                    if result:
                        file_path, entry = result
                        if file_path not in new_index["files"]:
                            new_index["files"][file_path] = entry
                        else:
                            new_index["files"][file_path].update(entry)

            # Update statistics
            new_index["statistics"].update({
                "total_files": len(new_index["files"]),
                "total_image_latents": sum(1 for f in new_index["files"].values() if "image_latent_path" in f),
                "total_text_latents": sum(1 for f in new_index["files"].values() if "text_latent_path" in f)
            })

            # Update cache index and save
            self.cache_index = new_index
            self._save_index()

            logger.info(
                f"Cache index rebuilt with {new_index['statistics']['total_files']} files "
                f"({new_index['statistics']['total_image_latents']} image latents, "
                f"{new_index['statistics']['total_text_latents']} text latents)"
            )

        except Exception as e:
            logger.error(f"Failed to rebuild cache index: {e}")
            raise

    def verify_cache_structure(self) -> bool:
        """Verify that cache files exist."""
        try:
            for file_path, info in self.cache_index["files"].items():
                # Just check if files exist
                if "image_latent_path" in info:
                    if not Path(info["image_latent_path"]).exists():
                        return False
                        
                if "text_latent_path" in info:
                    if not Path(info["text_latent_path"]).exists():
                        return False

            return True
            
        except Exception as e:
            logger.error(f"Error verifying cache structure: {str(e)}")
            return False
