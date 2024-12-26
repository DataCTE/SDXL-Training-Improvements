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
                base_name = Path(file_path).stem
                latent_path = self.image_latents_dir / f"{base_name}.pt"
                text_path = self.text_latents_dir / f"{base_name}.pt"
                
                latent_future = executor.submit(validate_file, latent_path)
                text_future = executor.submit(validate_file, text_path)
                
                valid_latent = latent_future.result()
                valid_text = text_future.result()
                
                if not valid_latent:
                    missing_latents.append(file_path)
                if not valid_text:
                    missing_text.append(file_path)
                if valid_latent or valid_text:
                    valid_files[file_path] = info

        self.cache_index["files"] = valid_files
        self._save_index()
        
        return missing_text, missing_latents

    def _load_index(self):
        try:
            with open(self.index_path) as f:
                self.cache_index = json.load(f)
        except:
            self.cache_index = {"files": {}}
            self._save_index()

    def _save_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)

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

            if image_latent is not None:
                image_latent = self._process_latents(image_latent, "image")
                latent_path = self.image_latents_dir / f"{base_name}.pt"
                torch.save({
                    "latent": image_latent,
                    "metadata": metadata
                }, latent_path)

            if text_latent is not None:
                text_latent = self._process_latents(text_latent.get("embeddings", {}), "text")
                text_path = self.text_latents_dir / f"{base_name}.pt"
                torch.save({
                    "embeddings": text_latent,
                    "metadata": {**metadata, "type": "text_latent"}
                }, text_path)

            self.cache_index["files"][str(file_path)] = {
                "base_name": base_name,
                "timestamp": time.time()
            }
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
                if "latent" in data:
                    data["latent"] = self._process_latents(data["latent"], "cached_latent")
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