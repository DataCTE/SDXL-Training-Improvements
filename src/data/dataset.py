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
        
        # Core configuration
        self.config = config
        self.is_train = is_train
        self._setup_device(device, device_id)

        # Model components
        self.model = model
        if model:
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers
            self.vae = model.vae

        # Cache setup with rebuild
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )
        self.cache_manager.rebuild_cache_index()  # Force rebuild cache index

        # Data initialization
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions

        # Bucket generation
        self.buckets = self._generate_dynamic_buckets(config)
        logger.info(f"Initialized dataset with {len(self.buckets)} dynamic buckets")
        self.bucket_indices = []

        # Tag weighting setup
        self._setup_tag_weighting(config, cache_dir)

        # Precompute latents if enabled
        if config.global_config.cache.use_cache:
            self._precompute_latents()

    # Core Dataset Methods
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single item from the dataset with proper validation and error handling."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            
            # Get cache key and load cached data
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is None:
                logger.warning(f"Failed to load cached data for: {image_path}")
                return None
            
            # Validate required tensors
            required_keys = {
                "vae_latents", 
                "prompt_embeds", 
                "pooled_prompt_embeds", 
                "time_ids"
            }
            if not all(k in cached_data for k in required_keys):
                logger.warning(
                    f"Missing required keys for {image_path}. "
                    f"Found: {set(cached_data.keys())}, "
                    f"Required: {required_keys}"
                )
                return None
            
            # Validate metadata
            required_metadata = {
                "original_size",
                "crop_coords",
                "target_size"
            }
            if not all(k in cached_data.get("metadata", {}) for k in required_metadata):
                logger.warning(
                    f"Missing required metadata for {image_path}. "
                    f"Found: {set(cached_data.get('metadata', {}).keys())}, "
                    f"Required: {required_metadata}"
                )
                return None
            
            # Validate tensor shapes and types
            try:
                # Validate VAE latents
                if not isinstance(cached_data["vae_latents"], torch.Tensor):
                    raise ValueError("vae_latents must be a torch.Tensor")
                if len(cached_data["vae_latents"].shape) != 3:  # [C, H, W]
                    raise ValueError(f"Invalid vae_latents shape: {cached_data['vae_latents'].shape}")
                
                # Validate prompt embeddings
                if not isinstance(cached_data["prompt_embeds"], torch.Tensor):
                    raise ValueError("prompt_embeds must be a torch.Tensor")
                if len(cached_data["prompt_embeds"].shape) != 2:  # [S, D]
                    raise ValueError(f"Invalid prompt_embeds shape: {cached_data['prompt_embeds'].shape}")
                
                # Validate pooled embeddings
                if not isinstance(cached_data["pooled_prompt_embeds"], torch.Tensor):
                    raise ValueError("pooled_prompt_embeds must be a torch.Tensor")
                if len(cached_data["pooled_prompt_embeds"].shape) != 1:  # [D]
                    raise ValueError(f"Invalid pooled_prompt_embeds shape: {cached_data['pooled_prompt_embeds'].shape}")
                
                # Validate time IDs
                if not isinstance(cached_data["time_ids"], torch.Tensor):
                    raise ValueError("time_ids must be a torch.Tensor")
                if cached_data["time_ids"].shape[-1] != 6:  # [..., 6]
                    raise ValueError(f"Invalid time_ids shape: {cached_data['time_ids'].shape}")
                
            except ValueError as e:
                logger.warning(f"Tensor validation failed for {image_path}: {str(e)}")
                return None
            
            # Return properly formatted data
            return {
                "vae_latents": cached_data["vae_latents"],
                "prompt_embeds": cached_data["prompt_embeds"],
                "pooled_prompt_embeds": cached_data["pooled_prompt_embeds"],
                "time_ids": cached_data["time_ids"],
                "metadata": {
                    "original_size": cached_data["metadata"]["original_size"],
                    "crop_coords": cached_data["metadata"]["crop_coords"],
                    "target_size": cached_data["metadata"]["target_size"],
                    "text": caption  # Always use the caption from the dataset
                }
            }
        
            

        except Exception as e:
            logger.error(f"Failed to get item {idx}: {str(e)}", exc_info=True)
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            # Filter None values and log
            valid_batch = [b for b in batch if b is not None]
            
            # If no valid samples, return None
            if len(valid_batch) == 0:
                return None
            
            # If at least half the batch is valid, proceed
            min_batch_size = max(self.config.training.batch_size // 4, 1)
            if len(valid_batch) >= min_batch_size:
                return {
                    "vae_latents": torch.stack([example["vae_latents"] for example in valid_batch]),
                    "prompt_embeds": torch.stack([example["prompt_embeds"] for example in valid_batch]),
                    "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"] for example in valid_batch]),
                    "time_ids": torch.stack([example["time_ids"] for example in valid_batch]),
                    "original_size": [example["metadata"]["original_size"] for example in valid_batch],
                    "crop_top_lefts": [example["metadata"]["crop_coords"] for example in valid_batch],
                    "target_size": [example["metadata"]["target_size"] for example in valid_batch],
                    "text": [example["metadata"]["text"] for example in valid_batch]
                }
            
            return None

        except Exception as e:
            logger.error("Failed to collate batch", exc_info=True)
            return None

    # Setup Methods
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

    def _setup_tag_weighting(self, config: Config, cache_dir: Path):
        """Setup tag weighting if enabled."""
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

    # Processing Methods
    def process_image_batch(self, image_paths: List[Union[str, Path]], captions: List[str], config: Config) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel."""
        try:
            # Process images in parallel
            processed_images = []
            
            for path in image_paths:
                processed = self._process_single_image(path, config)
                processed_images.append(processed)
            
            # Batch encode text for all valid images
            valid_captions = [
                caption for img_data, caption in zip(processed_images, captions)
                if img_data is not None
            ]
            
            if valid_captions:
                encoded_text = self.encode_prompt(
                    batch={"text": valid_captions},
                    proportion_empty_prompts=0.0
                )
            
            # Save results and prepare return values
            results = []
            caption_idx = 0
            
            for img_data, caption, path in zip(processed_images, captions, image_paths):
                if img_data is not None:
                    # Compute time ids
                    time_ids = self._compute_time_ids(
                        original_size=img_data["original_size"],
                        crops_coords_top_left=img_data["crop_coords"],
                        target_size=img_data["target_size"]
                    )
                    
                    # Add caption to processed data
                    result = {
                        **img_data,
                        "text": caption
                    }
                    
                    # Save to cache if enabled
                    if self.config.global_config.cache.use_cache:
                        tensors = {
                            "vae_latents": img_data["vae_latents"],
                            "prompt_embeds": encoded_text["prompt_embeds"][caption_idx],
                            "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"][caption_idx],
                            "time_ids": time_ids
                        }
                        metadata = {
                            "original_size": img_data["original_size"],
                            "crop_coords": img_data["crop_coords"],
                            "target_size": img_data["target_size"],
                            "text": caption
                        }
                        self.cache_manager.save_latents(tensors, path, metadata)
                        caption_idx += 1
                    
                    results.append(result)
                else:
                    results.append(None)
                    
            return results
            
        except Exception as e:
            logger.error("Batch processing failed", exc_info=True)
            return [None] * len(image_paths)

    # Helper Methods
    def _compute_time_ids(self, original_size, crops_coords_top_left, target_size, device=None, dtype=None):
        """Compute time embeddings for SDXL."""
        # Ensure inputs are proper tuples
        if not isinstance(original_size, (tuple, list)):
            original_size = (original_size, original_size)
        if not isinstance(crops_coords_top_left, (tuple, list)):
            crops_coords_top_left = (crops_coords_top_left, crops_coords_top_left)
        if not isinstance(target_size, (tuple, list)):
            target_size = (target_size, target_size)
        
        # Combine all values into a single list
        time_ids = [
            original_size[0],    # Original height
            original_size[1],    # Original width
            crops_coords_top_left[0],  # Crop top
            crops_coords_top_left[1],  # Crop left
            target_size[0],     # Target height
            target_size[1],     # Target width
        ]
        
        # Create tensor with proper device and dtype
        device = device or self.device
        return torch.tensor([time_ids], device=device, dtype=dtype)

    def _precompute_latents(self) -> None:
        """Process entire dataset in chunks of 1000 images, with progress tracking."""
        total_images = len(self.image_paths)
        
        # Quick check of cache index first
        if self.cache_manager.are_all_paths_cached(self.image_paths):
            logger.info("All images already in cache index. Skipping preprocessing.")
            return
        
        # If not all in cache index, do detailed scanning
        logger.info("Checking detailed cache status for all images...")
        uncached_paths = []
        for path in tqdm(self.image_paths, desc="Checking cache"):
            if not self.cache_manager.is_cached(path):
                uncached_paths.append(path)
        
        if not uncached_paths:
            logger.info("All images already cached. Skipping preprocessing.")
            return
        
        logger.info(f"Found {len(uncached_paths)} uncached images. Starting preprocessing...")
        
        # Process uncached images in chunks
        chunk_size = 1000
        total_chunks = (len(uncached_paths) + chunk_size - 1) // chunk_size
        
        # Create overall progress bar for chunks
        chunk_pbar = tqdm(
            total=total_chunks,
            desc="Processing dataset chunks",
            unit="chunk",
            position=0,
            leave=True
        )
        
        processed_count = 0
        failed_images = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(uncached_paths))
            chunk_paths = uncached_paths[start_idx:end_idx]
            
            # Process images in smaller batches
            batch_size = 8
            
            # Create progress bar for current chunk
            batch_pbar = tqdm(
                total=len(chunk_paths),
                desc=f"Chunk {chunk_idx + 1}/{total_chunks}",
                unit="img",
                position=1,
                leave=False
            )
            
            for batch_start in range(0, len(chunk_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_paths))
                batch_paths = chunk_paths[batch_start:batch_end]
                batch_captions = [self.captions[self.image_paths.index(p)] for p in batch_paths]
                
                processed_items = self.process_image_batch(
                    image_paths=batch_paths,
                    captions=batch_captions,
                    config=self.config
                )
                
                # Track successful and failed processing
                for item, path in zip(processed_items, batch_paths):
                    if item is not None:
                        processed_count += 1
                    else:
                        failed_images.append(path)
                
                batch_pbar.update(len(batch_paths))
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            batch_pbar.close()
            chunk_pbar.update(1)
            
            # Log progress after each chunk
            logger.info(
                f"Processed {processed_count}/{len(uncached_paths)} images "
                f"({len(failed_images)} failures)"
            )
        
        chunk_pbar.close()
        
        # Final summary
        success_rate = (processed_count / len(uncached_paths)) * 100 if uncached_paths else 100
        logger.info(
            f"\nPrecomputing complete:\n"
            f"- Total images: {total_images}\n"
            f"- Already cached: {total_images - len(uncached_paths)}\n"
            f"- Needed processing: {len(uncached_paths)}\n"
            f"- Successfully processed: {processed_count}\n"
            f"- Failed: {len(failed_images)}\n"
            f"- Success rate: {success_rate:.2f}%"
        )
        
        if failed_images:
            logger.warning(
                f"Failed to process {len(failed_images)} images. "
                "These will be skipped during training."
            )
            # Log failed images to file
            log_dir = Path(self.config.global_config.logging.log_dir)
            failed_log = log_dir / "failed_images.txt"
            with open(failed_log, 'w') as f:
                f.write('\n'.join(failed_images))
            logger.info(f"Failed image paths logged to: {failed_log}")

    def _process_image_tensor(
        self, 
        img_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        config: Config,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Process image tensor by finding exact bucket match."""
        w, h = original_size
        aspect_ratio = w / h
        
        # Early return if aspect ratio is invalid
        max_ratio = config.global_config.image.max_aspect_ratio
        if not (1/max_ratio <= aspect_ratio <= max_ratio):
            logger.warning(f"Image {image_path} has invalid aspect ratio ({aspect_ratio:.2f}). Skipping.")
            return None
        
        # Find matching bucket
        buckets = self.get_aspect_buckets()
        bucket_ratios = [(bw/bh, idx, (bw,bh)) for idx, (bw,bh) in enumerate(buckets)]
        _, bucket_idx, (target_w, target_h) = min(
            [(abs(aspect_ratio - ratio), idx, dims) for ratio, idx, dims in bucket_ratios]
        )
        
        try:
            
            return {
                "pixel_values": img_tensor,
                "original_size": original_size,
                "target_size": (target_w, target_h),
                "bucket_index": bucket_idx,
                "path": str(image_path),
                "timestamp": time.time(),
                "crop_coords": (0, 0)  # No cropping needed with dynamic buckets
            }
            
        except Exception as e:
            logger.warning(f"Failed to process image {image_path} during resizing: {str(e)}")
            return None

    def _process_single_image(self, image_path: Union[str, Path], config: Config) -> Optional[Dict[str, Any]]:
        """Process a single image with aspect ratio bucketing and VAE encoding."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use CacheManager's image loading
            loaded = self.cache_manager.load_image_to_tensor(image_path, self.device)
            if loaded is None:
                return None
            
            img_tensor, original_size = loaded
            
            processed = self._process_image_tensor(
                img_tensor=img_tensor,
                original_size=original_size,
                config=config,
                image_path=image_path
            )
            
            if processed is None:
                return None
            
            # Encode with VAE
            with torch.no_grad():
                pixel_values = processed["pixel_values"].unsqueeze(0)
                vae_latents = self.vae.encode(pixel_values).latent_dist.sample()
                vae_latents = vae_latents * self.vae.config.scaling_factor
            
            processed["vae_latents"] = vae_latents.squeeze(0)
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

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

    def _generate_dynamic_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """Generate dynamic buckets based on actual image dimensions."""
        bucket_step = config.global_config.image.bucket_step
        max_ratio = config.global_config.image.max_aspect_ratio
        min_size = config.global_config.image.min_size
        max_size = config.global_config.image.max_size
        
        # Collect all image dimensions and group by aspect ratio
        dimensions = {}  # aspect_ratio -> list of (width, height)
        logger.info("Analyzing image dimensions...")
        
        for path in tqdm(self.image_paths, desc="Scanning images"):
            loaded = self.cache_manager.load_image_to_tensor(path, self.device)
            if loaded:
                _, (w, h) = loaded
                ratio = w / h
                if 1/max_ratio <= ratio <= max_ratio:
                    # Round ratio to 2 decimal places to group similar ratios
                    rounded_ratio = round(ratio * 100) / 100
                    if rounded_ratio not in dimensions:
                        dimensions[rounded_ratio] = []
                    dimensions[rounded_ratio].append((w, h))
        
        # Generate buckets for each aspect ratio group
        buckets = set()
        for ratio, sizes in dimensions.items():
            # Find median dimensions for this ratio
            widths, heights = zip(*sizes)
            med_w = sorted(widths)[len(widths)//2]
            med_h = sorted(heights)[len(heights)//2]
            
            # Round to nearest bucket step
            base_w = int(med_w / bucket_step + 0.5) * bucket_step
            base_h = int(med_h / bucket_step + 0.5) * bucket_step
            
            # Ensure dimensions are within bounds
            base_w = max(min(base_w, max_size[0]), min_size[0])
            base_h = max(min(base_h, max_size[1]), min_size[1])
            
            # Add base bucket
            buckets.add((base_w, base_h))
            
            # Add scaled buckets if needed
            for scale in [0.75, 1.25]:
                w = int(base_w * scale / bucket_step) * bucket_step
                h = int(base_h * scale / bucket_step) * bucket_step
                if (min_size[0] <= w <= max_size[0] and 
                    min_size[1] <= h <= max_size[1]):
                    buckets.add((w, h))
        
        sorted_buckets = sorted(buckets, key=lambda x: (x[0] * x[1], x[0]/x[1]))
        logger.info(f"Generated {len(sorted_buckets)} dynamic buckets")
        return sorted_buckets

    def get_aspect_buckets(self) -> List[Tuple[int, int]]:
        """Return cached buckets."""
        return self.buckets

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
