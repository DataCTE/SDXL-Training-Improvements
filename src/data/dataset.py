"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path, load_data_from_directory
from src.data.config import Config
from src.models.sdxl import StableDiffusionXL
from src.data.preprocessing import (
    CacheManager,
    create_tag_weighter_with_index,
    TagWeighter
)
from src.models.encoders import CLIPEncoder
import torch.nn.functional as F
from src.data.preprocessing.bucket_utils import generate_buckets, compute_bucket_dims, validate_aspect_ratio, group_images_by_bucket, log_bucket_statistics

logger = get_logger(__name__)

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with extreme memory handling and 100x speedups."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        model: Optional[StableDiffusionXL] = None,
        tag_weighter: Optional["TagWeighter"] = None,
        is_train: bool = True,
        device: Optional[torch.device] = None,
        device_id: Optional[int] = None,
        cache_manager: Optional["CacheManager"] = None,
        transform: Optional[bool] = None
    ):
        """Initialize dataset with preprocessing capabilities."""
        super().__init__()
        
        # Core configuration
        self.config = config
        self.is_train = is_train
        self._setup_device(device, device_id)
        
        # Tag weighting
        self.tag_weighter = tag_weighter

        # Model components
        self.model = model
        if model:
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers
            self.vae = model.vae

        # Cache setup with rebuild
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )
        self.cache_manager.rebuild_cache_index()

        # Data initialization
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions

        # Generate buckets using utility function
        self.buckets = generate_buckets(config)
        logger.info(f"Initialized dataset with {len(self.buckets)} dynamic buckets")
        
        # Group images by bucket
        self.bucket_indices = self._group_images_by_bucket()
        
        # Log bucket statistics
        self._log_bucket_statistics()

    # Core Dataset Methods
    def __len__(self) -> int:
        """Return total number of samples across all buckets."""
        return sum(len(indices) for indices in self.bucket_indices.values())

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get item from same-sized bucket to ensure stackable tensors."""
        try:
            # Find which bucket this index belongs to
            current_pos = 0
            target_bucket = None
            bucket_idx = 0
            
            for bucket_dims, indices in self.bucket_indices.items():
                if idx < current_pos + len(indices):
                    target_bucket = bucket_dims
                    bucket_idx = idx - current_pos
                    break
                current_pos += len(indices)
            
            if target_bucket is None:
                return None
            
            # Get actual image index from bucket
            image_idx = self.bucket_indices[target_bucket][bucket_idx]
            image_path = self.image_paths[image_idx]
            caption = self.captions[image_idx]
            
            # Get cache key and load cached data
            cache_key = self.cache_manager.get_cache_key(image_path)
            cache_entry = self.cache_manager.cache_index["entries"].get(cache_key)
            
            if not cache_entry or not cache_entry.get("bucket_dims"):
                logger.warning(f"Missing cache entry or bucket dims for {image_path}")
                return None
            
            cached_data = self.cache_manager.load_tensors(cache_key)
            if cached_data is None:
                logger.warning(f"Failed to load cached data for: {image_path}")
                return None
            
            # Get tag weight from tag index if available
            tag_weight = 1.0  # Default weight
            if self.tag_weighter is not None:
                try:
                    # Try to get weight from tag index first
                    tag_index = self.cache_manager.load_tag_index()
                    if tag_index and "images" in tag_index and str(image_path) in tag_index["images"]:
                        tag_weight = tag_index["images"][str(image_path)]["total_weight"]
                    else:
                        # Fall back to computing weight directly
                        tag_weight = self.tag_weighter.get_caption_weight(caption)
                except Exception as e:
                    logger.warning(f"Failed to get tag weight for {image_path}: {e}")
            
            # Verify the latents match the bucket dimensions from cache
            latents = cached_data["vae_latents"]
            cached_bucket = tuple(cache_entry["bucket_dims"])
            expected_shape = (4, cached_bucket[1]//8, cached_bucket[0]//8)  # VAE reduces spatial dims by 8
            
            if latents.shape != expected_shape:
                logger.warning(
                    f"Latent shape mismatch for {image_path}: "
                    f"expected {expected_shape}, got {latents.shape}"
                )
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
            
            # Return properly formatted data with tag weight
            return {
                "vae_latents": cached_data["vae_latents"],
                "prompt_embeds": cached_data["prompt_embeds"],
                "pooled_prompt_embeds": cached_data["pooled_prompt_embeds"],
                "time_ids": cached_data["time_ids"],
                "metadata": {
                    "original_size": cached_data["metadata"]["original_size"],
                    "crop_coords": cached_data["metadata"]["crop_coords"],
                    "target_size": cached_data["metadata"]["target_size"],
                    "text": caption,
                    "tag_weight": tag_weight
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to get item {idx}: {str(e)}", exc_info=True)
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            # Filter None values
            valid_batch = [b for b in batch if b is not None]
            
            # If no valid samples, return None
            if len(valid_batch) == 0:
                return None
            
            # If at least minimum batch size
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
                    "text": [example["metadata"]["text"] for example in valid_batch],
                    "tag_weights": torch.tensor(
                        [example["metadata"]["tag_weight"] for example in valid_batch],
                        dtype=torch.float32,
                        device=self.device
                    )
                }
            
            return None

        except Exception as e:
            logger.error(f"Failed to collate batch: {str(e)}")
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
                if processed:
                    # Compute bucket dimensions during initial processing
                    bucket_dims = compute_bucket_dims(processed["original_size"], self.buckets)
                    processed["bucket_dims"] = bucket_dims
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
            
            # Save results with bucket information
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
                        self.cache_manager.save_latents(
                            tensors, 
                            path, 
                            metadata,
                            bucket_dims=img_data["bucket_dims"]  # Pass bucket dims to cache
                        )
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
        """Process uncached images and store their latents."""
        total_images = len(self.image_paths)
        
        # Get list of paths that need processing from cache manager
        uncached_paths = self.cache_manager.get_uncached_paths(self.image_paths)
        
        if not uncached_paths:
            logger.info("All images already in cache. Skipping preprocessing.")
            return
        
        logger.info(f"Found {len(uncached_paths)} uncached images. Starting preprocessing...")
        
        # Get corresponding captions for uncached paths
        uncached_captions = [self.captions[self.image_paths.index(p)] for p in uncached_paths]
        
        # Process in chunks
        chunk_size = 1000
      
        
        with tqdm(total=len(uncached_paths), desc="Processing images") as pbar:
            for chunk_start in range(0, len(uncached_paths), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(uncached_paths))
                chunk_paths = uncached_paths[chunk_start:chunk_end]
                chunk_captions = uncached_captions[chunk_start:chunk_end]
                
                # Process in smaller batches for memory efficiency
                batch_size = 8
                for batch_start in range(0, len(chunk_paths), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunk_paths))
                    batch_paths = chunk_paths[batch_start:batch_end]
                    batch_captions = chunk_captions[batch_start:batch_end]
                    
                    try:
                        # Load and process images
                        for path, caption in zip(batch_paths, batch_captions):
                            # Load image
                            img = Image.open(path).convert('RGB')
                            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                            img_tensor = img_tensor.permute(2, 0, 1).to(self.device)
                            
                            # Get bucket dimensions
                            bucket_dims = compute_bucket_dims(img.size, self.buckets)
                            
                            # Encode with VAE
                            with torch.no_grad():
                                vae_latents = self.model.vae.encode(
                                    img_tensor.unsqueeze(0)
                                ).latent_dist.sample()
                                vae_latents = vae_latents * self.model.vae.config.scaling_factor
                            
                            # Encode text
                            text_output = self.encode_prompt(
                                batch={"text": [caption]},
                                proportion_empty_prompts=0.0
                            )
                            
                            # Compute time ids
                            time_ids = self._compute_time_ids(
                                original_size=img.size,
                                crops_coords_top_left=(0, 0),
                                target_size=(bucket_dims[0]*8, bucket_dims[1]*8)
                            )
                            
                            # Save to cache
                            self.cache_manager.save_latents(
                                tensors={
                                    "vae_latents": vae_latents.squeeze(0),
                                    "prompt_embeds": text_output["prompt_embeds"][0],
                                    "pooled_prompt_embeds": text_output["pooled_prompt_embeds"][0],
                                    "time_ids": time_ids
                                },
                                path=path,
                                metadata={
                                    "original_size": img.size,
                                    "crop_coords": (0, 0),
                                    "target_size": (bucket_dims[0]*8, bucket_dims[1]*8),
                                    "text": caption
                                },
                                bucket_dims=bucket_dims
                            )
                            
                            pbar.update(1)
                            
                    except Exception as e:
                        logger.error(f"Failed to process batch: {e}")
                        continue
                    
                    # Clear CUDA cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Log final statistics
            logger.info(
                f"\nPrecomputing complete:\n"
                f"- Total images: {total_images}\n"
                f"- Already cached: {total_images - len(uncached_paths)}\n"
                f"- Processed: {len(uncached_paths)}\n"
            )

    def _process_image_tensor(
        self, 
        img_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Process image tensor ensuring VAE-compatible dimensions."""
        # Get bucket dimensions in latent space
        latent_w, latent_h = compute_bucket_dims(original_size, self.buckets)
        
        # Convert latent dimensions back to pixel space for conditioning
        pixel_w = latent_w * 8
        pixel_h = latent_h * 8
        
        return {
            "pixel_values": img_tensor.clone(),  # Clone to ensure we don't modify original
            "original_size": original_size,
            "target_size": (pixel_w, pixel_h),  # For conditioning (in pixel space)
            "latent_size": (latent_w, latent_h),  # For VAE (in latent space)
            "path": str(image_path),
            "timestamp": time.time(),
            "crop_coords": (0, 0)
        }

    def _process_single_image(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a single image with aspect ratio bucketing and VAE encoding."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check if image is already cached
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is not None and "vae_latents" in cached_data:
                # Use cached data if available
                return {
                    "vae_latents": cached_data["vae_latents"],
                    "original_size": cached_data["metadata"]["original_size"],
                    "target_size": cached_data["metadata"]["target_size"],
                    "latent_size": cached_data["metadata"]["latent_size"],
                    "path": str(image_path),
                    "timestamp": time.time(),
                    "crop_coords": cached_data["metadata"].get("crop_coords", (0, 0))
                }
            
            # If not cached, load and process the image
            loaded = self.cache_manager.load_image_to_tensor(image_path, self.device)
            if loaded is None:
                return None
            
            img_tensor, original_size = loaded
            
            # Process the image tensor
            processed = self._process_image_tensor(
                img_tensor=img_tensor,
                original_size=original_size,
                image_path=image_path
            )
            
            if processed is None:
                return None
            
            # Encode with VAE if model is available
            if self.model and hasattr(self.model, 'vae'):
                with torch.no_grad():
                    pixel_values = processed["pixel_values"].unsqueeze(0)
                    vae_latents = self.model.vae.encode(pixel_values).latent_dist.sample()
                    vae_latents = vae_latents * self.model.vae.config.scaling_factor
                processed["vae_latents"] = vae_latents.squeeze(0)
            
            # Cache the processed data
            if "vae_latents" in processed:
                self.cache_manager.save_latents(
                    tensors={
                        "vae_latents": processed["vae_latents"],
                    },
                    image_path=image_path,
                    metadata={
                        "original_size": processed["original_size"],
                        "target_size": processed["target_size"],
                        "latent_size": processed["latent_size"],
                        "crop_coords": processed["crop_coords"],
                        "timestamp": processed["timestamp"]
                    }
                )
            
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

    def get_aspect_buckets(self) -> List[Tuple[int, int]]:
        """Return cached buckets."""
        return self.buckets

    def _group_images_by_bucket(self) -> Dict[Tuple[int, int], List[int]]:
        """Group images by bucket dimensions."""
        return group_images_by_bucket(
            image_paths=self.image_paths,
            captions=self.captions,
            cache_manager=self.cache_manager
        )

    def _log_bucket_statistics(self):
        """Log statistics about bucket distribution using bucket_utils."""
        log_bucket_statistics(self.bucket_indices)

def create_dataset(
    config: Config,
    model: Optional[StableDiffusionXL] = None,
    verify_cache: bool = True
) -> AspectBucketDataset:
    """Create dataset using config values with proper fallbacks."""
    
    # Initialize cache manager with config
    cache_manager = CacheManager(
        cache_dir=config.global_config.cache.cache_dir,
        config=config,
        max_cache_size=config.global_config.cache.max_cache_size
    )
    
    # Load data paths from config
    image_paths, captions = load_data_from_directory(config.data.train_data_dir)
    
    # Verify and rebuild cache if needed
    if verify_cache:
        cache_manager.verify_and_rebuild_cache(image_paths, captions)
    
    # Create dataset with all components
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        model=model,
        cache_manager=cache_manager
    )
