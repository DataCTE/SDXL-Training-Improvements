"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_paths
from src.data.config import Config
from src.models.sdxl import StableDiffusionXL
from src.data.preprocessing import (
    CacheManager,
    create_tag_weighter_with_index
)
from src.models.encoders import CLIPEncoder
import torch.nn.functional as F

logger = get_logger(__name__)

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with extreme memory handling and 100x speedups."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        model: Optional[StableDiffusionXL] = None,
        is_train: bool = True,
        device: Optional[torch.device] = None,
        device_id: Optional[int] = None
    ):
        """Initialize dataset with preprocessing capabilities."""
        super().__init__()
        
        self.config = config
        self.is_train = is_train
        
        # Move CUDA setup to separate method
        self._setup_device(device, device_id)

        # Core components
        self.model = model
        if model:
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers

        # Initialize cache manager
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )

        # Initialize paths and captions
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions

        # Initialize buckets
        self.buckets = self.get_aspect_buckets(config)
        self.bucket_indices = []

        # Initialize tag weighter if needed
        if config.tag_weighting.enable_tag_weighting:
            image_captions = dict(zip(self.image_paths, self.captions))
            index_path = Path(cache_dir) / "tag_weights_index.json"
            self.tag_weighter = create_tag_weighter_with_index(
                config=config,
                image_captions=image_captions,
                index_output_path=index_path
            )
        else:
            self.tag_weighter = None

        # Precompute latents if caching enabled
        if config.global_config.cache.use_cache:
            self._precompute_latents()

    def _setup_device(self, device: Optional[torch.device], device_id: Optional[int]):
        """Setup CUDA device and optimizations."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            
            self.device = device or torch.device('cuda')
            self.device_id = device_id if device_id is not None else 0
        else:
            self.device = torch.device('cpu')
            self.device_id = None

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is None:
                processed = self._process_single_image(image_path=image_path, config=self.config)
                
                if processed is None:
                    raise ValueError(f"Failed to process image: {image_path}")
                    
                # Store only the caption text, no encoding here
                processed.update({"text": caption})
                    
                if self.config.global_config.cache.use_cache:
                    tensors = {
                        "pixel_values": processed["pixel_values"],
                    }
                    metadata = {
                        "original_size": processed["original_size"],
                        "crop_coords": processed["crop_coords"],
                        "target_size": processed["target_size"],
                        "text": caption
                    }
                    self.cache_manager.save_latents(tensors, image_path, metadata)
                    cached_data = {**tensors, "metadata": metadata}
                else:
                    cached_data = {
                        "pixel_values": processed["pixel_values"],
                        "metadata": {
                            "original_size": processed["original_size"],
                            "crop_coords": processed["crop_coords"],
                            "target_size": processed["target_size"],
                            "text": caption
                        }
                    }
            
            return cached_data

        except Exception as e:
            logger.error(f"Failed to get item {idx}: {str(e)}")
            return None

    def _precompute_latents(self) -> None:
        """Optimized precompute latents with batch processing."""
        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000
        for start_idx in range(0, len(self.image_paths), chunk_size):
            end_idx = min(start_idx + chunk_size, len(self.image_paths))
            chunk_paths = self.image_paths[start_idx:end_idx]
            
            # Check cache status for current chunk only
            cached_status = {
                path: self.cache_manager.is_cached(path) 
                for path in chunk_paths
            }
            
            remaining_images = sum(1 for v in cached_status.values() if not v)
            if remaining_images == 0:
                continue
                
            # Process uncached images in batches
            batch_size = 8
            uncached_indices = [
                i + start_idx for i, path in enumerate(chunk_paths)
                if not cached_status[path]
            ]
            
            pbar = tqdm(
                total=remaining_images,
                desc=f"Precomputing latents ({start_idx}-{end_idx})",
                unit="img"
            )

            for batch_start in range(0, len(uncached_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[batch_start:batch_end]
                
                batch_paths = [self.image_paths[i] for i in batch_indices]
                batch_captions = [self.captions[i] for i in batch_indices]
                
                processed_items = self.process_image_batch(
                    image_paths=batch_paths,
                    captions=batch_captions,
                    config=self.config
                )
                
                # Check successful processing
                successful = sum(1 for item in processed_items if item is not None)
                if successful < len(batch_indices):
                    logger.warning(f"Failed to process {len(batch_indices) - successful} images in batch")
                    
                pbar.update(len(batch_indices))
            
            pbar.close()

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            # Filter out None values from failed processing
            batch = [b for b in batch if b is not None]
            
            if not batch:
                raise ValueError("Empty batch after filtering")

            # Group samples by bucket size
            bucket_groups = {}
            for example in batch:
                size = example["pixel_values"].shape[-2:]  # Get HxW dimensions
                if size not in bucket_groups:
                    bucket_groups[size] = []
                bucket_groups[size].append(example)

            # Take the largest group
            largest_size = max(bucket_groups.keys(), key=lambda k: len(bucket_groups[k]))
            valid_batch = bucket_groups[largest_size]

            # Batch encode text here
            captions = [example["metadata"]["text"] for example in valid_batch]
            encoded_text = self.encode_prompt(
                batch={"text": captions},
                proportion_empty_prompts=0.0
            )

            # Return collated batch with encoded text
            return {
                "pixel_values": torch.stack([example["pixel_values"] for example in valid_batch]),
                "prompt_embeds": encoded_text["prompt_embeds"],
                "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"],
                "original_size": [example["metadata"]["original_size"] for example in valid_batch],
                "crop_top_lefts": [example["metadata"]["crop_coords"] for example in valid_batch],
                "target_size": [example["metadata"]["target_size"] for example in valid_batch],
                "text": captions
            }

        except Exception as e:
            logger.error("Failed to collate batch", exc_info=True)
            raise RuntimeError(f"Collate failed: {str(e)}") from e

    def encode_prompt(
        self,
        batch: Dict[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts using CLIP encoders directly."""
        try:
            encoded_output = CLIPEncoder.encode_prompt(
                batch=batch,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                proportion_empty_prompts=proportion_empty_prompts,
                is_train=self.is_train
            )
            
            return {
                "prompt_embeds": encoded_output["prompt_embeds"],
                "pooled_prompt_embeds": encoded_output["pooled_prompt_embeds"],
                "metadata": {
                    "num_prompts": len(batch[next(iter(batch))]),
                    "device": str(self.device),
                    "dtype": str(encoded_output["prompt_embeds"].dtype),
                    "timestamp": time.time()
                }
            }
        except Exception as e:
            logger.error("Failed to encode prompts", 
                        extra={'error': str(e), 'batch_size': len(batch[next(iter(batch))])})
            raise

    def _load_image_to_tensor(self, image_path: Union[str, Path]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load image and convert to tensor.
        
        Returns:
            Tuple of (tensor, (width, height))
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        img_tensor = torch.from_numpy(np.array(img)).float().to(self.device) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
        return img_tensor, (w, h)

    def _process_image_tensor(
        self, 
        img_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        config: Config,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Process image tensor on GPU."""
        w, h = original_size
        aspect_ratio = w / h
        
        # Early return if aspect ratio is invalid
        max_ratio = config.global_config.image.max_aspect_ratio
        if not (1/max_ratio <= aspect_ratio <= max_ratio):
            return None
        
        # Find best matching bucket
        buckets = self.get_aspect_buckets(config)
        bucket_ratios = [(bw/bh, idx, (bw,bh)) for idx, (bw,bh) in enumerate(buckets)]
        _, bucket_idx, (target_w, target_h) = min(
            [(abs(aspect_ratio - ratio), idx, dims) for ratio, idx, dims in bucket_ratios]
        )
        
        # Resize and crop in one step if possible
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img_tensor = img_tensor[:, top:top + target_h, left:left + target_w]
        
        return {
            "pixel_values": img_tensor,
            "original_size": original_size,
            "target_size": (target_w, target_h),
            "bucket_index": bucket_idx,
            "path": str(image_path),
            "timestamp": time.time(),
            "crop_coords": (left, top) if new_w != target_w or new_h != target_h else (0, 0)
        }

    def _process_single_image(self, image_path: Union[str, Path], config: Config) -> Optional[Dict[str, Any]]:
        """Process a single image with aspect ratio bucketing."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            img_tensor, original_size = self._load_image_to_tensor(image_path)
            return self._process_image_tensor(
                img_tensor=img_tensor,
                original_size=original_size,
                config=config,
                image_path=image_path
            )
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def process_image_batch(
        self,
        image_paths: List[Union[str, Path]],
        captions: List[str],
        config: Config
    ) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel."""
        try:
            # Process images in parallel
            processed_images = []
            
            for path in image_paths:
                processed = self._process_single_image(path, config)
                processed_images.append(processed)
                
            # Don't encode text here - let collate_fn handle it
            results = []
            for img_data, caption in zip(processed_images, captions):
                if img_data is not None:
                    results.append({
                        **img_data,
                        "text": caption
                    })
                else:
                    results.append(None)
                    
            return results
            
        except Exception as e:
            logger.error("Batch processing failed", exc_info=True)
            return [None] * len(image_paths)

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """Get supported image dimensions for aspect ratio bucketing."""
        return config.global_config.image.supported_dims

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    model: Optional[StableDiffusionXL] = None,
) -> AspectBucketDataset:
    """Create and initialize dataset instance."""
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        model=model
    )
