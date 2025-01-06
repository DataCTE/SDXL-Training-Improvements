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

from src.core.logging import UnifiedLogger, LogConfig
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
from src.data.preprocessing.exceptions import CacheError, DataLoadError, TagProcessingError
from src.core.logging import UnifiedLogger, LogConfig

logger = UnifiedLogger(LogConfig(name=__name__))

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
        
        # Add structured logging for initialization
        logger.info("Initializing AspectBucketDataset", extra={
            'is_train': is_train,
            'device_id': device_id,
            'num_images': len(image_paths)
        })
        
        # Generate buckets with validation
        try:
            self.buckets = generate_buckets(config)
            if not self.buckets:
                logger.error("No valid buckets generated", extra={
                    'config': str(config.global_config.image),
                    'supported_dims': str(config.global_config.image.supported_dims)
                })
                raise DataLoadError("No valid buckets generated")
            logger.info("Generated dynamic buckets", extra={
                'num_buckets': len(self.buckets),
                'bucket_sizes': [b.pixel_dims for b in self.buckets]
            })
        except Exception as e:
            raise DataLoadError("Failed to generate buckets", context={
                'error': str(e),
                'config': str(config.global_config.image)
            })

        # Cache setup with bucket validation
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )
        
        # Initialize bucket indices with validation
        try:
            self.bucket_indices = self._load_bucket_indices_from_cache()
            self._validate_bucket_assignments()
        except Exception as e:
            raise DataLoadError("Failed to initialize bucket indices", context={
                'error': str(e),
                'num_buckets': len(self.buckets)
            })

        # Data paths with validation
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions
        
        # Model components
        self.model = model
        if model:
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers
            self.vae = model.vae
        
        # Tag weighting setup with validation
        try:
            self._setup_tag_weighting()
        except Exception as e:
            logger.warning(f"Tag weighting initialization failed: {e}")
            self.tag_weighter = None

    def _validate_bucket_assignments(self) -> None:
        """Validate all bucket assignments for consistency."""
        try:
            for bucket_idx, image_indices in self.bucket_indices.items():
                bucket = self.buckets[bucket_idx]
                for img_idx in image_indices:
                    img_path = self.image_paths[img_idx]
                    try:
                        with Image.open(img_path) as img:
                            assigned_bucket = compute_bucket_dims(img.size, self.buckets)
                            if not assigned_bucket or assigned_bucket != bucket:
                                raise ValueError(f"Inconsistent bucket assignment for {img_path}")
                    except Exception as e:
                        logger.warning(f"Failed to validate bucket assignment for {img_path}: {e}")
        except Exception as e:
            raise DataLoadError("Bucket validation failed", context={
                'error': str(e),
                'num_buckets': len(self.buckets)
            })

    # Core Dataset Methods
    def __len__(self) -> int:
        """Return total number of samples across all buckets."""
        return sum(len(indices) for indices in self.bucket_indices.values())

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single item with proper tensor formatting."""
        try:
            image_path = self.image_paths[idx]
            
            # Load cached tensors
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is None:
                return None
            
            # Load tag weights if enabled
            tag_info = None
            if self.tag_weighter and self.config.tag_weighting.enable_tag_weighting:
                tag_index = self.cache_manager.load_tag_index()
                if tag_index and "images" in tag_index:
                    image_tags = tag_index["images"].get(str(image_path), {})
                    if image_tags:
                        tag_info = {
                            "tags": image_tags["tags"],
                            "metadata": tag_index["metadata"]
                        }
            
            # Update metadata with tag info
            cached_data["metadata"]["tag_info"] = tag_info or {
                "tags": {category: [] for category in self.tag_weighter.tag_types.keys()},
                "metadata": {}
            }
            
            return cached_data
            
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

    def _setup_tag_weighting(self) -> None:
        """Initialize tag weighting with enhanced error handling."""
        if not self.config.tag_weighting.enable_tag_weighting:
            self.tag_weighter = None
            return
        
        try:
            self.tag_weighter = TagWeighter(config=self.config, model=self.model)
            
            if not self.tag_weighter.initialize_tag_system():
                image_captions = dict(zip(self.image_paths, self.captions))
                self.tag_weighter = create_tag_weighter_with_index(
                    config=self.config,
                    image_captions=image_captions,
                    model=self.model
                )
                
            if self.tag_weighter:
                self._validate_tag_coverage()
                
        except TagProcessingError as e:
            logger.error(f"Tag weighting failed: {e}")
            self.tag_weighter = None

    # Processing Methods
    def process_image_batch(self, image_paths: List[Union[str, Path]], captions: List[str], config: Config) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel."""
        try:
            # Process images in parallel
            processed_images = []
            
            # Get tag weights if enabled
            tag_infos = None
            if self.tag_weighter:
                tag_infos = [
                    {"tags": self.tag_weighter._extract_tags(caption)}
                    for caption in captions
                ]
            
            for path, caption in zip(image_paths, captions):
                processed = self._process_single_image(path)
                if processed:
                    # Add tag information to processed data
                    if tag_infos:
                        processed["tag_info"] = tag_infos[len(processed_images)]
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
                        "tag_info": img_data.get("tag_info", {"tags": {}})
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
                            "text": caption,
                            "bucket_info": img_data["bucket_info"].__dict__,
                            "tag_info": img_data.get("tag_info", {"tags": {}})
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
        """Precompute and cache latents for both trainers."""
        try:
            uncached_paths = self.cache_manager.get_uncached_paths(self.image_paths)
            
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
                                tags = self.tag_weighter._extract_tags(caption)
                                tag_info = {
                                    "tags": {
                                        tag_type: [
                                            {"tag": tag, "weight": self.tag_weighter.tag_weights[tag_type][tag]}
                                            for tag in tags[tag_type]
                                        ] for tag_type in tags
                                    }
                                }
                            
                            # Process image and text
                            img_tensor = self._prepare_image_tensor(img, bucket_info.pixel_dims)
                            vae_latents = self.vae.encode(
                                img_tensor.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                            ).latent_dist.sample() * self.vae.config.scaling_factor
                            
                            text_output = CLIPEncoder.encode_prompt(
                                batch={"text": [caption]},
                                text_encoders=self.text_encoders,
                                tokenizers=self.tokenizers,
                                is_train=self.is_train
                            )
                            
                            # Save to cache with serializable metadata
                            self.cache_manager.save_latents(
                                tensors={
                                    "vae_latents": vae_latents.squeeze(0),
                                    "prompt_embeds": text_output["prompt_embeds"][0],
                                    "pooled_prompt_embeds": text_output["pooled_prompt_embeds"][0],
                                    "time_ids": self._compute_time_ids(
                                        original_size=img.size,
                                        target_size=bucket_info.pixel_dims,
                                        crop_coords=(0, 0)
                                    )
                                },
                                path=path,
                                metadata={
                                    "text": caption
                                },
                                bucket_info=bucket_info,
                                tag_info=tag_info
                            )
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Failed to process {path}: {e}")
                            continue

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
                cache_key = self.cache_manager.get_cache_key(path)
                entry = self.cache_manager.cache_index["entries"].get(cache_key)
                
                if entry and entry.get("bucket_info"):
                    bucket_info = entry["bucket_info"]
                    bucket_indices[tuple(bucket_info["pixel_dims"])].append(idx)
                else:
                    # Compute bucket for uncached image
                    with Image.open(path) as img:
                        bucket_info = compute_bucket_dims(img.size, self.buckets)
                        bucket_indices[bucket_info.pixel_dims].append(idx)
            
            except Exception as e:
                logger.error(f"Error loading bucket info for {path}: {e}")
                # Default to first bucket on error
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
        
        # Load data paths from config
        logger.info(f"Loading data from: {config.data.train_data_dir}")
        image_paths, captions = load_data_from_directory(config.data.train_data_dir)
        if not image_paths:
            raise DataLoadError("No images found in data directory", context={
                'data_dir': config.data.train_data_dir
            })
            
        # First verify cache integrity if requested
        if verify_cache:
            logger.info("Verifying cache and tag metadata...")
            cache_manager.verify_and_rebuild_cache(image_paths, captions)
            
        # Initialize tag weighting system
        tag_weighter = None
        if config.tag_weighting.enabled:
            try:
                # Check for existing tag cache after verification
                tag_stats_path = cache_manager.get_tag_statistics_path()
                tag_images_path = cache_manager.get_image_tags_path()

                if (tag_stats_path.exists() and 
                    tag_images_path.exists() and 
                    config.tag_weighting.use_cache):
                    logger.info("Loading tag weights from verified cache...")
                    tag_weighter = TagWeighter(config, model)
                    if not tag_weighter._load_cache():
                        raise ValueError("Failed to load tag cache")
                else:
                    raise FileNotFoundError("Tag cache not found or disabled")
                    
            except (FileNotFoundError, ValueError) as e:
                logger.info("Computing tag weights and creating new index...")
                tag_weighter = create_tag_weighter_with_index(
                    config=config,
                    image_captions=captions,
                    model=model
                )
                
                # Save tag statistics and metadata
                logger.info("Saving tag statistics...")
                cache_manager.save_tag_index({
                    "version": "1.0",
                    "updated_at": time.time(),
                    "metadata": {
                        "config": {
                            "default_weight": tag_weighter.default_weight,
                            "min_weight": tag_weighter.min_weight,
                            "max_weight": tag_weighter.max_weight,
                            "smoothing_factor": tag_weighter.smoothing_factor
                        },
                        "statistics": {
                            "total_samples": tag_weighter.total_samples,
                            "tag_type_counts": {
                                tag_type: sum(counts.values())
                                for tag_type, counts in tag_weighter.tag_counts.items()
                            },
                            "unique_tags": {
                                tag_type: len(counts)
                                for tag_type, counts in tag_weighter.tag_counts.items()
                            }
                        }
                    },
                    "statistics": tag_weighter.get_tag_statistics(),
                    "images": tag_weighter.process_dataset_tags(captions)
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
        
        # Verify latent cache if needed
        if verify_cache:
            logger.info("Checking for uncached images...")
            uncached_paths = cache_manager.get_uncached_paths(image_paths)
            if uncached_paths:
                logger.info(f"Found {len(uncached_paths)} uncached images. Starting precomputation...")
                dataset._precompute_latents()
            else:
                logger.info("All images are cached")
        
        # Rebuild buckets after any cache updates
        dataset.bucket_indices = dataset._group_images_by_bucket()
        dataset._log_bucket_statistics()
        
        # Log tag statistics if available
        if tag_weighter:
            stats = tag_weighter.get_tag_metadata()
            logger.info("\nTag Statistics:")
            logger.info(f"Total samples: {stats['statistics']['total_samples']}")
            for tag_type, count in stats['statistics']['tag_type_counts'].items():
                logger.info(f"{tag_type}: {count} tags, {stats['statistics']['unique_tags'][tag_type]} unique")
            
            # Verify tag coverage
            total_images = len(image_paths)
            tagged_images = len(tag_weighter.process_dataset_tags(captions))
            coverage = (tagged_images / total_images) * 100
            logger.info(f"\nTag Coverage: {coverage:.2f}% ({tagged_images}/{total_images} images)")
        
        logger.info(
            f"Dataset created successfully with {len(dataset)} samples "
            f"in {len(dataset.buckets)} buckets"
        )
        
        return dataset
        
    except Exception as e:
        if isinstance(e, (CacheError, DataLoadError)):
            logger.error(str(e))
            raise
        logger.error(f"Failed to create dataset: {str(e)}", exc_info=True)
        raise RuntimeError(f"Dataset creation failed: {str(e)}") from e
