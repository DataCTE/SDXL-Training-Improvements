"""High-performance cache management for large-scale dataset preprocessing."""
import multiprocessing as mp
from src.core.logging.logging import setup_logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import json
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from src.core.memory.tensor import (
    torch_gc,
    pin_tensor_,
    unpin_tensor_,
    tensors_to_device_,
    create_stream_context,
    tensors_record_stream,
    device_equals
)
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)

logger = setup_logging(__name__, level="INFO")

class CacheManager:
    """Manages high-throughput caching of image-caption pairs."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        num_proc: Optional[int] = None,
        chunk_size: int = 1000,
        compression: Optional[str] = "zstd",
        verify_hashes: bool = True
    ):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cached files
            num_proc: Number of processes (default: CPU count)
            chunk_size: Number of items per cache chunk
            compression: Compression algorithm (None, 'zstd', 'gzip')
            verify_hashes: Whether to verify content hashes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_proc = num_proc or mp.cpu_count()
        self.chunk_size = chunk_size
        self.compression = compression
        self.verify_hashes = verify_hashes
        
        # Setup cache index
        self.index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Setup process pools
        self.image_pool = ProcessPoolExecutor(max_workers=self.num_proc)
        self.io_pool = ThreadPoolExecutor(max_workers=self.num_proc * 2)
        
    def _load_cache_index(self) -> Dict:
        """Load or create cache index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"files": {}, "chunks": {}}
        
    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f)
            
    def _compute_hash(self, file_path: Path) -> str:
        """Compute file hash for verification."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def _process_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Process single image with optimized memory handling and CUDA streams."""
        try:
            # Use NVIDIA DALI for faster image loading if available
            try:
                import nvidia.dali as dali
                import nvidia.dali.fn as fn
                use_dali = True
            except ImportError:
                use_dali = False
                
            if use_dali:
                pipe = dali.Pipeline(batch_size=1, num_threads=1, device_id=0)
                with pipe:
                    images = fn.readers.file(name="Reader", files=[str(image_path)])
                    decoded = fn.decoders.image(images, device="mixed")
                    normalized = fn.crop_mirror_normalize(
                        decoded,
                        dtype=dali.types.FLOAT,
                        mean=[0.5 * 255] * 3,
                        std=[0.5 * 255] * 3,
                        output_layout="CHW"
                    )
                    pipe.set_outputs(normalized)
                pipe.build()
                tensor = pipe.run()[0].as_tensor()
            else:
                # Fallback to PIL with optimizations
                image = Image.open(image_path).convert('RGB')
                tensor = torch.from_numpy(np.array(image)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # CHW format
                
            # Optimize memory format
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            # Pin memory if CUDA is available
            if torch.cuda.is_available():
                tensor = tensor.pin_memory()
                
            return tensor
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
            
    def _save_chunk(
        self,
        chunk_id: int,
        tensors: List[torch.Tensor],
        metadata: Dict
    ) -> bool:
        """Save processed chunk to disk."""
        try:
            chunk_path = self.cache_dir / f"chunk_{chunk_id:06d}.pt"
            
            # Save tensors with optional compression
            if self.compression == "zstd":
                torch.save(tensors, chunk_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(tensors, chunk_path)
                
            # Update index
            self.cache_index["chunks"][str(chunk_id)] = {
                "path": str(chunk_path),
                "size": len(tensors),
                "metadata": metadata
            }
            
            return True
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_id}: {str(e)}")
            return False
            
    def process_dataset(
        self,
        data_dir: Union[str, Path],
        image_exts: List[str] = [".jpg", ".jpeg", ".png"],
        caption_ext: str = ".txt", 
        num_workers: Optional[int] = None
    ) -> Dict[str, int]:
        """Process dataset with performance metrics logging.
        
        Args:
            data_dir: Directory containing image-caption pairs
            image_exts: List of valid image extensions
            caption_ext: Caption file extension
            num_workers: Number of worker processes
            
        Returns:
            Dict with processing statistics
        """
        logger.info(f"Starting dataset processing with {num_workers or self.num_proc} workers")
        from src.utils.paths import convert_windows_path, is_wsl
        data_dir = convert_windows_path(data_dir, make_absolute=True)
        if is_wsl():
            logger.info(f"Running in WSL, using converted path: {data_dir}")
        """Process entire dataset with parallel processing.
        
        Args:
            data_dir: Directory containing image-caption pairs
            image_exts: List of valid image extensions
            caption_ext: Caption file extension
            
        Returns:
            Processing statistics
        """
        data_dir = Path(data_dir)
        stats = {"processed": 0, "failed": 0, "skipped": 0}
        
        # Get all image files with WSL path handling
        image_files = []
        for ext in image_exts:
            found_files = list(data_dir.glob(f"*{ext}"))
            # Convert any Windows paths
            image_files.extend([
                Path(str(convert_windows_path(f, make_absolute=True)))
                for f in found_files
            ])
            
        logger.info(f"Found {len(image_files)} images to process")
        
        # Use provided num_workers or default
        workers = num_workers if num_workers is not None else self.num_proc
        
        # Process in chunks
        for chunk_start in tqdm(range(0, len(image_files), self.chunk_size), 
                               desc="Processing chunks",
                               disable=workers > 1):
            chunk = image_files[chunk_start:chunk_start + self.chunk_size]
            chunk_id = chunk_start // self.chunk_size
            
            # Process images in parallel
            futures = []
            for img_path in chunk:
                if str(img_path) in self.cache_index["files"]:
                    stats["skipped"] += 1
                    continue
                    
                caption_path = img_path.with_suffix(caption_ext)
                if not caption_path.exists():
                    stats["failed"] += 1
                    continue
                    
                futures.append(
                    self.image_pool.submit(
                        self._process_image, 
                        img_path
                    ) if workers > 1 else self._process_image(img_path)
                )
                
            # Collect results
            tensors = []
            metadata = {}
            
            for i, future in enumerate(futures):
                try:
                    tensor = future.result()
                    if tensor is not None:
                        img_path = chunk[i]
                        caption_path = img_path.with_suffix(caption_ext)
                        
                        # Read caption
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                            
                        # Store tensor and metadata
                        tensors.append(tensor)
                        metadata[str(img_path)] = {
                            "caption": caption,
                            "hash": self._compute_hash(img_path) if self.verify_hashes else None
                        }
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Error processing chunk item: {str(e)}")
                    stats["failed"] += 1
                    
            # Save chunk if not empty
            if tensors:
                if self._save_chunk(chunk_id, tensors, metadata):
                    # Update main index
                    for img_path in metadata:
                        self.cache_index["files"][img_path] = {
                            "chunk_id": chunk_id,
                            "metadata": metadata[img_path]
                        }
                        
            # Save index periodically
            if chunk_id % 10 == 0:
                self._save_cache_index()
                
        # Final index save
        self._save_cache_index()
        
        return stats
        
    def get_cached_item(
        self,
        image_path: Union[str, Path]
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """Retrieve cached item by image path."""
        image_path = str(Path(image_path))
        
        if image_path not in self.cache_index["files"]:
            return None
            
        file_info = self.cache_index["files"][image_path]
        chunk_id = file_info["chunk_id"]
        chunk_info = self.cache_index["chunks"][str(chunk_id)]
        
        # Load chunk
        chunk_path = Path(chunk_info["path"])
        if not chunk_path.exists():
            return None
            
        tensors = torch.load(chunk_path)
        caption = file_info["metadata"]["caption"]
        
        return tensors, caption
        
    def verify_cache(self) -> Dict[str, int]:
        """Verify cache integrity."""
        stats = {"valid": 0, "corrupted": 0, "missing": 0}
        
        for file_path, file_info in tqdm(self.cache_index["files"].items()):
            if not self.verify_hashes:
                continue
                
            try:
                current_hash = self._compute_hash(file_path)
                stored_hash = file_info["metadata"]["hash"]
                
                if current_hash == stored_hash:
                    stats["valid"] += 1
                else:
                    stats["corrupted"] += 1
            except FileNotFoundError:
                stats["missing"] += 1
                
        return stats
        
    def clear_cache(self):
        """Clear all cached files."""
        # Remove chunk files
        for chunk_info in self.cache_index["chunks"].values():
            try:
                Path(chunk_info["path"]).unlink()
            except FileNotFoundError:
                pass
                
        # Clear index
        self.cache_index = {"files": {}, "chunks": {}}
        self._save_cache_index()
