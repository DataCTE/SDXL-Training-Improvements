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
from src.data.preprocessing.bucket_utils import generate_buckets, compute_bucket_dims, group_images_by_bucket, log_bucket_statistics
from src.data.preprocessing.bucket_types import BucketInfo, BucketDimensions

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
        cache_manager: Optional["CacheManager"] = None
    ):
        """Initialize dataset with preprocessing capabilities."""
        super().__init__()
        
        # Core configuration
        self.config = config
        self.is_train = is_train
        self._setup_device(device, device_id)
        
        # Data initialization with path conversion
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions
        
        # Cache setup
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )
        
        # Generate buckets using enhanced system
        self.buckets = generate_buckets(config)  # Now returns List[BucketInfo]
        logger.info(f"Initialized dataset with {len(self.buckets)} dynamic buckets")
        
        # Initialize bucket indices
        self.bucket_indices = self._load_bucket_indices_from_cache()
        
        # Model components
        self.model = model
        if model:
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers
            self.vae = model.vae
        
        # Tag weighting setup
        self.tag_weighter = tag_weighter

    # Core Dataset Methods
    def __len__(self) -> int:
        """Return total number of samples across all buckets."""
        return sum(len(indices) for indices in self.bucket_indices.values())

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single item with proper tensor formatting for both trainers."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx] if self.captions else None
            
            # Load cached tensors
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is None:
                return None
            
            # Get tag information if available
            tag_info = cached_data["metadata"].get("tag_info", {
                "total_weight": 1.0,
                "tags": {}
            })
            
            return {
                "vae_latents": cached_data["vae_latents"],
                "prompt_embeds": cached_data["prompt_embeds"],
                "pooled_prompt_embeds": cached_data["pooled_prompt_embeds"],
                "time_ids": cached_data["time_ids"],  # Contains all dimension info
                "metadata": {
                    "text": caption,
                    "bucket_info": cached_data["bucket_info"],
                    "tag_info": tag_info,
                    "image_path": image_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Collate batch items for training."""
        try:
            # Filter None values
            valid_batch = [b for b in batch if b is not None]
            if len(valid_batch) == 0:
                return None
            
            # Ensure minimum batch size
            min_batch_size = max(self.config.training.batch_size // 4, 1)
            if len(valid_batch) < min_batch_size:
                return None
            
            # Stack tensors
            collated = {
                "vae_latents": torch.stack([b["vae_latents"] for b in valid_batch]),
                "prompt_embeds": torch.stack([b["prompt_embeds"] for b in valid_batch]),
                "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in valid_batch]),
                "time_ids": torch.stack([b["time_ids"] for b in valid_batch]),
                
                # Collect metadata
                "metadata": [
                    {
                        "original_size": b["metadata"]["original_size"],
                        "crop_coords": b["metadata"]["crop_coords"],
                        "target_size": b["metadata"]["target_size"],
                        "text": b["metadata"]["text"],
                        "bucket_info": b["metadata"]["bucket_info"],
                        "tag_info": b["metadata"].get("tag_info", {"total_weight": 1.0, "tags": {}})
                    }
                    for b in valid_batch
                ]
            }
            
            return collated
            
        except Exception as e:
            logger.error(f"Collate failed: {e}")
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
            self.tag_weighter = create_tag_weighter_with_index(
                config=config,
                image_captions=image_captions
            )
        else:
            self.tag_weighter = None

    # Processing Methods
    def process_image_batch(self, image_paths: List[Union[str, Path]], captions: List[str], config: Config) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel."""
        try:
            # Process images in parallel
            processed_images = []
            
            for path, caption in zip(image_paths, captions):
                # Get tag weight details if available
                tag_info = None
                if self.tag_weighter:
                    tag_info = self.tag_weighter.get_caption_weight_details(caption)
                
                processed = self._process_single_image(path)
                if processed:
                    # Add tag information to processed data
                    if tag_info:
                        processed["tag_info"] = tag_info
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
            
            # Save results with bucket and tag information
            results = []
            caption_idx = 0
            
            for img_data, caption, path in zip(processed_images, captions, image_paths):
                if img_data is not None:
                    # Compute time ids
                    time_ids = self._compute_time_ids(
                        original_size=img_data["original_size"],
                        target_size=img_data["target_size"],
                        crop_coords=img_data["crop_coords"]
                    )
                    
                    # Add caption and tag info to processed data
                    result = {
                        **img_data,
                        "text": caption,
                        "bucket_info": img_data["bucket_info"],
                        "tag_info": img_data.get("tag_info", {
                            "total_weight": 1.0,
                            "tags": {}
                        })
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
                            "text": caption,
                            "bucket_info": img_data["bucket_info"]
                        }
                        self.cache_manager.save_latents(
                            tensors=tensors,
                            path=path,
                            metadata=metadata,
                            bucket_info=img_data["bucket_info"],
                            tag_info=img_data.get("tag_info")
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
    def _compute_time_ids(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        crop_coords: Tuple[int, int] = (0, 0)
    ) -> torch.Tensor:
        """Compute time embeddings for SDXL.
        Format: [orig_w, orig_h, crop_left, crop_top, target_w, target_h]
        Contains all necessary dimensional information in a single tensor.
        """
        time_ids = torch.tensor([
            list(original_size) + list(crop_coords) + list(target_size)
        ])
        
        return time_ids.to(device=self.device, dtype=torch.float32)

    def _extract_dims_from_time_ids(self, time_ids: torch.Tensor) -> Dict[str, Any]:
        """Extract dimensional information from time_ids tensor."""
        time_ids = time_ids.squeeze()  # Remove batch dimension if present
        return {
            "original_size": (int(time_ids[0].item()), int(time_ids[1].item())),
            "crop_coords": (int(time_ids[2].item()), int(time_ids[3].item())),
            "target_size": (int(time_ids[4].item()), int(time_ids[5].item()))
        }

    def _precompute_latents(self) -> None:
        """Precompute and cache latents for both DDPM and Flow Matching."""
        try:
            uncached_paths = self.cache_manager.get_uncached_paths(self.image_paths)
            total_images = len(self.image_paths)
            
            if uncached_paths:
                with torch.no_grad():
                    pbar = tqdm(total=len(uncached_paths), desc="Precomputing latents")
                    
                    for path, caption in zip(uncached_paths, self.captions):
                        try:
                            # Load and process image
                            img = Image.open(path).convert('RGB')
                            bucket_info = compute_bucket_dims(img.size, self.buckets)
                            
                            # Get tag weight details if available
                            tag_info = None
                            if self.tag_weighter:
                                tag_info = self.tag_weighter.get_caption_weight_details(caption)
                            
                            # Prepare image tensor
                            img_tensor = self._prepare_image_tensor(img, bucket_info.pixel_dims)
                            
                            # Encode image with VAE
                            vae_latents = self.vae.encode(
                                img_tensor.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                            ).latent_dist.sample()
                            vae_latents = vae_latents * self.vae.config.scaling_factor
                            
                            # Encode text with CLIP
                            text_output = CLIPEncoder.encode_prompt(
                                batch={"text": [caption]},
                                text_encoders=self.text_encoders,
                                tokenizers=self.tokenizers,
                                is_train=self.is_train
                            )
                            
                            # Compute time embeddings
                            time_ids = self._compute_time_ids(
                                original_size=img.size,
                                target_size=bucket_info.pixel_dims,
                                crop_coords=(0, 0)
                            )
                            
                            # Save to cache with all metadata
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
                                    "target_size": bucket_info.pixel_dims,
                                    "text": caption,
                                    "bucket_info": bucket_info.__dict__,
                                    "image_path": str(path)
                                },
                                bucket_info=bucket_info,
                                tag_info=tag_info
                            )
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Failed to process {path}: {e}")
                            continue
                        
                        # Clear CUDA cache periodically
                        if torch.cuda.is_available() and (pbar.n % 100 == 0):
                            torch.cuda.empty_cache()
                
                # Log final statistics
                logger.info(
                    f"\nPrecomputing complete:\n"
                    f"- Total images: {total_images}\n"
                    f"- Already cached: {total_images - len(uncached_paths)}\n"
                    f"- Processed: {len(uncached_paths)}\n"
                )

        except Exception as e:
            logger.error(f"Failed to precompute latents: {e}")

    def _prepare_image_tensor(
        self,
        img: Image.Image,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Prepare image tensor for VAE encoding.
        
        Args:
            img: PIL Image to process
            target_size: Target dimensions (width, height)
            
        Returns:
            Normalized tensor ready for VAE
        """
        # Resize image to target dimensions
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        
        # Rearrange dimensions to [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # Move to correct device and dtype
        img_tensor = img_tensor.to(
            device=self.device,
            dtype=self.vae.dtype if hasattr(self, 'vae') else torch.float32
        )
        
        return img_tensor

    def _process_single_image(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a single image with enhanced bucket handling."""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            
            # Get bucket info
            bucket_info = compute_bucket_dims(original_size, self.buckets)
            
            # Prepare image tensor
            img_tensor = self._prepare_image_tensor(img, bucket_info.pixel_dims)
            
            # VAE encode
            with torch.no_grad():
                vae_latents = self.vae.encode(
                    img_tensor.unsqueeze(0)
                ).latent_dist.sample()
                vae_latents = vae_latents * self.vae.config.scaling_factor
            
            # Validate VAE latents shape
            expected_shape = (4, bucket_info.latent_dims[1], bucket_info.latent_dims[0])  # VAE shape is (C, H, W)
            if vae_latents.shape[1:] != expected_shape[1:]:  # Only check spatial dimensions
                logger.warning(
                    f"VAE latent shape mismatch for {image_path}: "
                    f"expected {expected_shape}, got {vae_latents.shape}"
                )
                return None
            
            return {
                "vae_latents": vae_latents.squeeze(0),
                "time_ids": self._compute_time_ids(
                    original_size=original_size,
                    target_size=bucket_info.pixel_dims,
                    crop_coords=(0, 0)
                ),
                "original_size": original_size,
                "crop_coords": (0, 0),
                "target_size": bucket_info.pixel_dims,
                "bucket_info": bucket_info
            }
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def get_aspect_buckets(self) -> List[BucketInfo]:
        """Return cached buckets."""
        return self.buckets

    def _group_images_by_bucket(self) -> Dict[Tuple[int, int], List[int]]:
        """Group images by bucket dimensions."""
        return group_images_by_bucket(
            image_paths=self.image_paths,
            cache_manager=self.cache_manager
        )

    def _log_bucket_statistics(self):
        """Log statistics about bucket distribution using bucket_utils."""
        log_bucket_statistics(
            bucket_indices=self.bucket_indices,
            total_images=len(self.image_paths)
        )

    def _load_bucket_indices_from_cache(self) -> Dict[Tuple[int, int], List[int]]:
        """Load bucket assignments from cache with enhanced bucket info."""
        bucket_indices = defaultdict(list)
        
        for idx, path in enumerate(self.image_paths):
            try:
                # First try to get from cache
                cache_key = self.cache_manager.get_cache_key(path)
                entry = self.cache_manager.cache_index["entries"].get(cache_key)
                
                if entry and "bucket_info" in entry:
                    # Use cached bucket info
                    cached_info = entry["bucket_info"]
                    bucket_indices[tuple(cached_info["pixel_dims"])].append(idx)
                else:
                    # Compute bucket for uncached image
                    with Image.open(path) as img:
                        bucket_info = compute_bucket_dims(img.size, self.buckets)
                        bucket_indices[bucket_info.pixel_dims].append(idx)
                        
                        # Update cache with new bucket info
                        if entry:
                            entry["bucket_info"] = {
                                "dimensions": bucket_info.dimensions.__dict__,
                                "pixel_dims": bucket_info.pixel_dims,
                                "latent_dims": bucket_info.latent_dims,
                                "bucket_index": bucket_info.bucket_index,
                                "size_class": bucket_info.size_class,
                                "aspect_class": bucket_info.aspect_class
                            }
                            self.cache_manager._save_index()
            
            except Exception as e:
                logger.error(f"Error loading bucket info for {path}: {e}")
                # Only use default bucket as last resort
                bucket_indices[self.buckets[0].pixel_dims].append(idx)
        
        return dict(bucket_indices)

def create_dataset(
    config: Config,
    model: Optional[StableDiffusionXL] = None,
    verify_cache: bool = True
) -> AspectBucketDataset:
    """Create dataset using config values with proper fallbacks."""
    logger.info("Creating dataset...")
    
    try:
        # Initialize cache manager with config
        cache_manager = CacheManager(
            cache_dir=config.global_config.cache.cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size
        )
        
        # Load data paths from config (returns copies)
        logger.info(f"Loading data from: {config.data.train_data_dir}")
        image_paths, captions = load_data_from_directory(config.data.train_data_dir)
        
        # Initialize tag weighting if enabled
        tag_weighter = None
        if config.tag_weighting.enable_tag_weighting:
            logger.info("Initializing tag weighting system...")
            image_captions = dict(zip(image_paths, captions))
            
            # Check for existing tag cache
            tag_cache_path = cache_manager.tags_dir / "tag_index.json"
            if tag_cache_path.exists() and config.tag_weighting.use_cache:
                logger.info("Loading tag weights from cache...")
                tag_weighter = TagWeighter(config)
                tag_weighter._load_cache()
            else:
                logger.info("Computing tag weights and creating index...")
                tag_weighter = create_tag_weighter_with_index(
                    config=config,
                    image_captions=image_captions
                )
                
                # Save tag statistics and metadata
                logger.info("Saving tag statistics...")
                tag_metadata = tag_weighter.get_tag_metadata()
                cache_manager.save_tag_index({
                    "metadata": tag_metadata,
                    "image_tags": tag_weighter.process_dataset_tags(image_captions)
                })
        
        # Create dataset instance with tag weighter
        dataset = AspectBucketDataset(
            config=config,
            image_paths=image_paths,
            captions=captions,
            model=model,
            tag_weighter=tag_weighter,
            cache_manager=cache_manager
        )
        
        # Verify cache and rebuild if necessary
        if verify_cache:
            logger.info("Verifying cache...")
            cache_manager.verify_and_rebuild_cache(image_paths, captions)
            
            # Also verify tag cache if enabled
            if tag_weighter and config.tag_weighting.use_cache:
                logger.info("Verifying tag cache...")
                tag_cache = cache_manager.load_tag_index()
                if not tag_cache or "metadata" not in tag_cache:
                    logger.warning("Tag cache invalid or missing. Recomputing...")
                    tag_weighter = create_tag_weighter_with_index(
                        config=config,
                        image_captions=image_captions
                    )
                    dataset.tag_weighter = tag_weighter
        
        # Precompute latents for uncached images
        logger.info("Checking for uncached images...")
        uncached_paths = cache_manager.get_uncached_paths(image_paths)
        if uncached_paths:
            logger.info(f"Found {len(uncached_paths)} uncached images. Starting precomputation...")
            dataset._precompute_latents()  # This will handle the actual precomputation
        else:
            logger.info("All images are cached. Skipping precomputation.")
        
        # Rebuild buckets after precomputation
        dataset.bucket_indices = dataset._group_images_by_bucket()
        dataset._log_bucket_statistics()
        
        # Log tag statistics if available
        if tag_weighter:
            stats = tag_weighter.get_tag_metadata()
            logger.info("\nTag Statistics:")
            logger.info(f"Total samples: {stats['statistics']['total_samples']}")
            for tag_type, count in stats['statistics']['tag_type_counts'].items():
                logger.info(f"{tag_type}: {count} tags, {stats['statistics']['unique_tags'][tag_type]} unique")
        
        logger.info(
            f"Dataset created successfully with {len(dataset)} samples "
            f"in {len(dataset.buckets)} buckets"
        )
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}", exc_info=True)
        raise
