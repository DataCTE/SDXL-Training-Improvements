"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
import weakref
import random

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from src.core.logging import UnifiedLogger, LogConfig, ProgressPredictor, get_logger
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
from src.data.preprocessing.bucket_utils import generate_buckets, compute_bucket_dims, group_images_by_bucket, log_bucket_statistics, verify_bucket_assignment
from src.data.preprocessing.bucket_types import BucketInfo, BucketDimensions
from src.data.preprocessing.exceptions import CacheError, DataLoadError, TagProcessingError
from src.core.logging import UnifiedLogger, LogConfig
from src.data.preprocessing.samplers import BucketBatchSampler

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
        
        # Initialize basic attributes
        self.config = config
        self.is_train = is_train
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = device_id
        
        # Initialize cache manager
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=config.global_config.cache.cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )
        
        self.image_paths = image_paths
        self.captions = captions
        
        # Generate buckets from config
        self.buckets = generate_buckets(self.config)
        
        # Initialize bucket assignments using bucket_utils
        self.bucket_indices = group_images_by_bucket(self.image_paths, self.cache_manager)
        
        # Store model components as weak references
        self._model = None
        self._text_encoders = None
        self._tokenizers = None
        self._vae = None
        self._tag_weighter = None
        
        if model is not None and not self._in_worker_process():
            self.initialize_model_components(model)
        
        if tag_weighter is not None and not self._in_worker_process():
            self._tag_weighter = tag_weighter
    
    @staticmethod
    def _in_worker_process() -> bool:
        """Check if we're in a worker process."""
        return hasattr(torch.utils.data.get_worker_info(), 'id')
    
    def initialize_model_components(self, model: StableDiffusionXL):
        """Initialize model components. Called in both main and worker processes."""
        self._model = model
        self._text_encoders = model.text_encoders
        self._tokenizers = model.tokenizers
        self._vae = model.vae
    
    @property
    def model(self) -> Optional[StableDiffusionXL]:
        return self._model
        
    @property
    def text_encoders(self):
        return self._text_encoders
        
    @property
    def tokenizers(self):
        return self._tokenizers
        
    @property
    def vae(self):
        return self._vae
        
    @property
    def tag_weighter(self):
        return self._tag_weighter
    
    @staticmethod
    def worker_init_fn(worker_id: int):
        """Initialize worker process with proper CUDA settings."""
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        
        try:
            # Set worker device
            if torch.cuda.is_available():
                device_id = worker_id % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                dataset.device = torch.device(f'cuda:{device_id}')
                dataset.device_id = device_id
            else:
                dataset.device = torch.device('cpu')
                dataset.device_id = None
            
            # Initialize cache manager for worker
            from src.data.preprocessing import CacheManager
            dataset.cache_manager = CacheManager(
                cache_dir=dataset.config.global_config.cache.cache_dir,
                config=dataset.config,
                max_cache_size=dataset.config.global_config.cache.max_cache_size,
                device=dataset.device
            )
            
            logger.debug(f"Worker {worker_id} initialized successfully on device {dataset.device}")
            
        except Exception as e:
            logger.error(f"Error in worker {worker_id} initialization: {str(e)}", exc_info=True)
            raise

    # Core Dataset Methods
    def __len__(self) -> int:
        """Return total number of samples across all buckets."""
        return sum(len(indices) for indices in self.bucket_indices.values())

    def __getitem__(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a single item from the dataset with proper error handling."""
        try:
            # Get cache key for this index
            cache_key = self.image_paths[index]
            
            # Load tensors with error handling
            try:
                tensors = self.cache_manager.load_tensors(cache_key)
                if tensors is None:
                    logger.warning(f"Failed to load tensors for index {index}, cache_key: {cache_key}")
                    return None
                    
                # Validate required keys
                required_keys = {
                    "vae_latents", "prompt_embeds", "pooled_prompt_embeds",
                    "time_ids", "metadata"
                }
                if not all(k in tensors for k in required_keys):
                    missing = required_keys - set(tensors.keys())
                    logger.warning(f"Missing required keys for index {index}: {missing}")
                    return None
                    
                return tensors
                
            except Exception as e:
                logger.warning(
                    f"Error loading tensors for index {index}:\n"
                    f"Cache key: {cache_key}\n"
                    f"Error: {str(e)}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {index}: {str(e)}", exc_info=True)
            return None

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        """Collate batch items for training with proper error handling."""
        try:
            # Filter out None values
            valid_batch = [b for b in batch if b is not None]
            
            # Return None if no valid items
            if not valid_batch:
                logger.warning("No valid items in batch")
                return None
                
            # Ensure all required keys are present
            required_keys = {
                "vae_latents", "prompt_embeds", "pooled_prompt_embeds",
                "time_ids", "metadata"
            }
            
            # Validate each item
            for item in valid_batch:
                if not all(k in item for k in required_keys):
                    missing = required_keys - set(item.keys())
                    logger.warning(f"Item missing required keys: {missing}")
                    return None
                    
            # Collate tensors
            try:
                collated = {
                    "vae_latents": torch.stack([b["vae_latents"] for b in valid_batch]),
                    "prompt_embeds": torch.stack([b["prompt_embeds"] for b in valid_batch]),
                    "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in valid_batch]),
                    "time_ids": torch.stack([b["time_ids"] for b in valid_batch]),
                    "metadata": [b["metadata"] for b in valid_batch]
                }
                
                # Validate tensor shapes
                expected_batch_size = len(valid_batch)
                for key, tensor in collated.items():
                    if key != "metadata" and tensor.shape[0] != expected_batch_size:
                        logger.warning(
                            f"Invalid batch size for {key}: "
                            f"got {tensor.shape[0]}, expected {expected_batch_size}"
                        )
                        return None
                        
                return collated
                
            except Exception as e:
                logger.warning(f"Error collating tensors: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in collate_fn: {str(e)}", exc_info=True)
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
                self.tag_weighter = create_tag_weighter_with_index(
                    config=self.config,
                    captions=self.captions,
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
            logger.info("Starting latent precomputation...")
            
            # Log start of uncached path check
            logger.info("Checking for uncached paths...")
            start_time = time.time()
            uncached_paths = self.cache_manager.get_uncached_paths(self.image_paths)
            check_time = time.time() - start_time
            logger.info(f"Found {len(uncached_paths)} uncached paths in {check_time:.2f}s")
            
            if uncached_paths:
                logger.info("Beginning latent precomputation for uncached images...")
                with torch.no_grad():
                    pbar = tqdm(
                        total=len(uncached_paths),
                        desc="Precomputing latents",
                        dynamic_ncols=True
                    )
                    
                    for path, caption in zip(uncached_paths, self.captions):
                        try:
                            process_start = time.time()
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
                            save_start = time.time()
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
                            process_time = time.time() - process_start
                            save_time = time.time() - save_start
                            
                            if process_time > 5:  # Log slow processing
                                logger.debug(f"Slow image processing: {path} took {process_time:.2f}s (save: {save_time:.2f}s)")
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Failed to process {path}: {e}")
                            continue
                            
                    pbar.close()
            else:
                logger.info("No uncached paths found, skipping precomputation")

        except Exception as e:
            logger.error(f"Failed to precompute latents: {e}", exc_info=True)

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

    def __getstate__(self):
        """Get picklable state, excluding model components."""
        state = self.__dict__.copy()
        # Remove unpicklable components
        state['_model'] = None
        state['_text_encoders'] = None
        state['_tokenizers'] = None
        state['_vae'] = None
        state['_tag_weighter'] = None
        state['cache_manager'] = None
        return state

    def __setstate__(self, state):
        """Restore state, model components will be reinitialized in worker."""
        self.__dict__.update(state)

    def ensure_model_components(self):
        """Ensure model components are properly initialized."""
        try:
            # Ensure device is set
            if not hasattr(self, 'device'):
                if torch.cuda.is_available():
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        device_id = worker_info.id % torch.cuda.device_count()
                    else:
                        device_id = 0
                    self.device = torch.device(f'cuda:{device_id}')
                    self.device_id = device_id
                else:
                    self.device = torch.device('cpu')
                    self.device_id = None
            
            # Ensure cache manager is initialized
            if not hasattr(self, 'cache_manager') or self.cache_manager is None:
                from src.data.preprocessing import CacheManager
                self.cache_manager = CacheManager(
                    cache_dir=self.config.global_config.cache.cache_dir,
                    config=self.config,
                    max_cache_size=self.config.global_config.cache.max_cache_size,
                    device=self.device
                )
                logger.debug(f"Initialized cache manager for worker on device {self.device}")
            
            # Initialize model components if needed
            if hasattr(self, '_model_ref'):
                model = self._model_ref()
                if model is not None:
                    # Get text encoders from weak references
                    self._text_encoders = [ref() for ref in self._text_encoders if ref() is not None]
                    if not self._text_encoders:
                        self._text_encoders = [encoder.to(self.device) for encoder in model.text_encoders]
                    
                    # Get VAE from weak reference
                    vae = self._vae()
                    if vae is None:
                        self._vae = weakref.ref(model.vae.to(self.device))
                    
                    logger.debug(f"Refreshed model components on device {self.device}")
            
            # Initialize tag weighter if needed
            if hasattr(self, '_tag_weighter') and self._tag_weighter is not None:
                if not hasattr(self._tag_weighter, 'cache_manager') or self._tag_weighter.cache_manager is None:
                    self._tag_weighter.cache_manager = self.cache_manager
                    self._tag_weighter.initialize_tag_system()
                    logger.debug("Initialized tag weighter system")
                    
        except Exception as e:
            logger.error(f"Error ensuring model components: {str(e)}", exc_info=True)
            raise

    def _group_images_by_bucket(self) -> Dict[Tuple[int, ...], List[int]]:
        """Group images by their bucket dimensions using bucket utils."""
        return group_images_by_bucket(self.image_paths, self.cache_manager)
    
    def _log_bucket_statistics(self):
        """Log bucket statistics using bucket utils."""
        log_bucket_statistics(self.bucket_indices, len(self.image_paths))

    def create_bucket_sampler(self, batch_size: int, drop_last: bool = True, shuffle: bool = True) -> BucketBatchSampler:
        """Create a bucket-aware batch sampler."""
        return BucketBatchSampler(
            bucket_indices=self.bucket_indices,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle
        )

def create_dataset(
    config: Config,
    model: Optional[StableDiffusionXL] = None,
    verify_cache: bool = True
) -> AspectBucketDataset:
    """Create dataset using config values."""
    # Initialize cache manager with config
    cache_manager = CacheManager(
        cache_dir=config.global_config.cache.cache_dir,
        config=config,
        max_cache_size=config.global_config.cache.max_cache_size
    )
    
    # Load data paths from config
    image_paths, captions = load_data_from_directory(config.data.train_data_dir)
    
    # Create dataset instance
    dataset = AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        model=model,
        tag_weighter=None,
        cache_manager=cache_manager
    )
    
    return dataset
