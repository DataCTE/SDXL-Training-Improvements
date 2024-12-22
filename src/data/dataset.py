"""High-performance dataset implementation for SDXL training with optimized memory handling."""
import os
import threading
import traceback
import time
import psutil
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager, nullcontext

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

from src.core.logging.logging import setup_logging
from src.core.memory.tensor import (
    create_stream_context,
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    torch_gc,
    tensors_to_device_,
    device_equals
)
from .utils.paths import convert_windows_path
from .config import Config
from src.data.preprocessing import (
    LatentPreprocessor, TagWeighter, create_tag_weighter,
    CacheManager, PreprocessingPipeline
)

logger = setup_logging(__name__)

class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class ImageLoadError(DatasetError):
    """Raised when image loading fails."""
    pass

class BucketingError(DatasetError):
    """Raised when bucketing operations fail."""
    pass

class ProcessingError(DatasetError):
    """Raised when tensor processing fails."""
    pass

@dataclass
class DatasetStats:
    """Track dataset statistics."""
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_allocated: int = 0
    peak_memory: int = 0

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with optimized memory handling and bucketing."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        tag_weighter: Optional[TagWeighter] = None,
        is_train: bool = True,
        enable_memory_tracking: bool = True,
        max_memory_usage: float = 0.8
    ):
        """Initialize dataset with enhanced memory management.
        
        Args:
            config: Training configuration
            image_paths: List of image paths
            captions: List of captions
            latent_preprocessor: Optional preprocessor for latents
            tag_weighter: Optional tag weighter
            is_train: Whether this is training data
            enable_memory_tracking: Whether to track memory usage
            max_memory_usage: Maximum fraction of GPU memory to use
        """
        super().__init__()

        # Initialize statistics tracking
        self.stats = DatasetStats()
        self.enable_memory_tracking = enable_memory_tracking
        self.max_memory_usage = max_memory_usage
        
        # Setup CUDA optimizations
        if torch.cuda.is_available():
            self._setup_cuda_optimizations()
            
        # Process and validate paths
        self.image_paths = self._validate_paths(image_paths)
        self.stats.total_images = len(self.image_paths)
        
        # Store configuration
        self.config = config
        self.captions = captions
        self.latent_preprocessor = latent_preprocessor
        self.tag_weighter = tag_weighter or self._create_tag_weighter(config, captions)
        self.is_train = is_train
        
        # Initialize preprocessing pipeline if latent preprocessor is available
        self.preprocessing_pipeline = (
            PreprocessingPipeline(
                config=config,
                latent_preprocessor=latent_preprocessor,
                enable_memory_tracking=enable_memory_tracking,
                use_pinned_memory=True
            ) if latent_preprocessor else None
        )
        
        # Setup image configuration
        self._setup_image_config()
        
        # Initialize cache manager if enabled
        self.cache_manager = self._setup_cache_manager(config)

        # Precompute latents if needed
        if latent_preprocessor and config.global_config.cache.use_cache:
            self._precompute_latents(
                image_paths,
                captions,
                latent_preprocessor,
                config
            )
        
        # Create buckets and assign indices
        self.buckets = self._create_buckets()
        self.bucket_indices = self._assign_buckets()
        
        # Setup transforms with memory optimization
        self.transforms = self._setup_transforms()

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True

    def _create_tag_weighter(self, config: Config, captions: List[str]) -> Optional[TagWeighter]:
        """Create tag weighter if enabled in config."""
        if hasattr(config, 'tag_weighting') and config.tag_weighting.enable_tag_weighting:
            return create_tag_weighter(config, captions)
        return None

    def _setup_image_config(self):
        """Setup image processing configuration."""
        self.target_size = tuple(map(int, self.config.global_config.image.target_size))
        self.max_size = tuple(map(int, self.config.global_config.image.max_size))
        self.min_size = tuple(map(int, self.config.global_config.image.min_size))
        self.bucket_step = int(self.config.global_config.image.bucket_step)
        self.max_aspect_ratio = float(self.config.global_config.image.max_aspect_ratio)

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms with memory optimization."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _validate_paths(self, paths: List[str]) -> List[str]:
        """Validate and convert image paths with error tracking."""
        validated_paths = []
        
        for path in paths:
            try:
                converted = convert_windows_path(path, make_absolute=True)
                if os.path.exists(str(converted)):
                    validated_paths.append(str(converted))
                else:
                    self.stats.failed_images += 1
                    logger.warning(f"Path does not exist: {converted}")
            except Exception as e:
                self.stats.failed_images += 1
                logger.error(f"Path validation failed: {str(e)}")
                
        if not validated_paths:
            raise DatasetError("No valid image paths found")
            
        return validated_paths

    def _precompute_latents(
        self,
        image_paths: List[str],
        captions: List[str],
        latent_preprocessor: LatentPreprocessor,
        config: Config
    ) -> None:
        """Precompute and cache latents for all images."""
        logger.info("Checking for cached latents...")
        
        # Get list of uncached images
        uncached_indices = []
        uncached_paths = []
        uncached_captions = []
        
        for idx, (img_path, caption) in enumerate(zip(image_paths, captions)):
            if not self.cache_manager.get_cached_item(img_path):
                uncached_indices.append(idx)
                uncached_paths.append(img_path)
                uncached_captions.append(caption)

        if not uncached_paths:
            logger.info("All latents are cached")
            return

        logger.info(f"Precomputing latents for {len(uncached_paths)} images...")
        
        # Process in batches
        batch_size = config.training.batch_size
        for i in tqdm(range(0, len(uncached_paths), batch_size), desc="Caching latents"):
            batch_paths = uncached_paths[i:i + batch_size]
            batch_captions = uncached_captions[i:i + batch_size]
            
            try:
                # Load and process images with detailed error handling
                images = []
                for img_path in batch_paths:
                    try:
                        # Validate path
                        if not Path(img_path).exists():
                            logger.error(f"Image not found: {img_path}")
                            continue
                            
                        # Check file size
                        file_size = Path(img_path).stat().st_size
                        if file_size == 0:
                            logger.error(f"Empty image file: {img_path}")
                            continue
                            
                        # Load image
                        try:
                            image = Image.open(img_path)
                        except Exception as e:
                            logger.error(f"Failed to open {img_path}: {str(e)}")
                            continue
                            
                        # Validate format
                        if image.format not in ('JPEG', 'PNG', 'WEBP'):
                            logger.error(f"Unsupported format {image.format} for {img_path}")
                            continue
                            
                        # Convert to RGB
                        try:
                            image = image.convert('RGB')
                        except Exception as e:
                            logger.error(f"RGB conversion failed for {img_path}: {str(e)}")
                            continue
                            
                        # Process image
                        try:
                            processed = self._process_image(
                                image,
                                self.target_size,
                                latent_preprocessor.vae.device
                            )
                            images.append(processed)
                        except Exception as e:
                            logger.error(
                                f"Processing failed for {img_path}: {str(e)}\n"
                                f"Image details:\n"
                                f"- Size: {image.size}\n"
                                f"- Mode: {image.mode}\n"
                                f"- Format: {image.format}\n"
                                f"- Is animated: {getattr(image, 'is_animated', False)}\n"
                                f"- Frames: {getattr(image, 'n_frames', 1)}\n"
                                f"- Memory usage: {image.size[0] * image.size[1] * len(image.getbands())} bytes\n"
                                f"- Bands: {image.getbands()}\n"
                                f"- Info: {image.info}\n"
                                f"Error context:\n"
                                f"- Error type: {type(e).__name__}\n"
                                f"- Stack trace:\n{traceback.format_exc()}"
                            )
                            continue
                            
                    except Exception as e:
                        logger.error(
                            f"Unexpected error with {img_path}:\n"
                            f"Error details:\n"
                            f"- Message: {str(e)}\n"
                            f"- Type: {type(e).__name__}\n"
                            f"- File stats:\n"
                            f"  - Size: {Path(img_path).stat().st_size} bytes\n"
                            f"  - Modified: {time.ctime(Path(img_path).stat().st_mtime)}\n"
                            f"  - Permissions: {oct(Path(img_path).stat().st_mode)[-3:]}\n"
                            f"System context:\n"
                            f"- Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB\n"
                            f"- Used GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB\n"
                            f"- Process memory: {psutil.Process().memory_info().rss/1e9:.2f}GB\n"
                            f"Stack trace:\n{traceback.format_exc()}"
                        )
                        continue

                if not images:
                    continue

                # Stack images and generate embeddings
                pixel_values = torch.stack(images)
                text_embeddings = latent_preprocessor.encode_prompt(
                    batch_captions,
                    proportion_empty_prompts=config.data.proportion_empty_prompts,
                    is_train=self.is_train
                )
                
                # Generate latents
                with torch.no_grad():
                    vae_output = latent_preprocessor.encode_images(pixel_values)
                
                # Cache results
                for j, img_path in enumerate(batch_paths):
                    if j >= len(images):
                        continue
                        
                    self.cache_manager.save_preprocessed_data(
                        latent_data={
                            "image": images[j],
                            "vae_latents": vae_output["model_input"][j:j+1]
                        },
                        text_embeddings={
                            "prompt_embeds": text_embeddings["prompt_embeds"][j:j+1],
                            "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"][j:j+1]
                        },
                        metadata={
                            "original_size": self.target_size,
                            "crop_top_left": (0, 0),
                            "target_size": self.target_size
                        },
                        file_path=img_path
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
                
            # Clean up GPU memory
            torch_gc()

        logger.info("Latent precomputation complete")

    def _create_buckets(self) -> List[Tuple[int, int]]:
        """Create aspect ratio buckets from config."""
        return self.config.global_config.image.supported_dims

    def _assign_buckets(self) -> List[int]:
        """Assign images to buckets based on aspect ratio."""
        bucket_indices = []
        
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path)
                w, h = img.size
                aspect_ratio = w / h
                img_area = w * h
                
                # Find best bucket match
                min_diff = float('inf')
                best_idx = 0
                
                for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
                    bucket_ratio = bucket_w / bucket_h
                    if bucket_ratio > self.max_aspect_ratio:
                        continue
                        
                    ratio_diff = abs(aspect_ratio - bucket_ratio)
                    area_diff = abs(img_area - (bucket_w * bucket_h))
                    total_diff = (ratio_diff * 2.0) + (area_diff / (1536 * 1536))
                    
                    if total_diff < min_diff:
                        min_diff = total_diff
                        best_idx = idx
                        
                bucket_indices.append(best_idx)
                
            except Exception as e:
                logger.error(f"Error assigning bucket for {img_path}: {str(e)}")
                bucket_indices.append(0)
                
        return bucket_indices

    def _setup_cache_manager(self, config: Config) -> Optional[CacheManager]:
        """Initialize cache manager with configuration."""
        if not config.global_config.cache.use_cache:
            return None
            
        try:
            cache_dir = Path(convert_windows_path(
                config.global_config.cache.cache_dir,
                make_absolute=True
            ))
            
            return CacheManager(
                cache_dir=cache_dir,
                num_proc=config.global_config.cache.num_proc,
                chunk_size=config.global_config.cache.chunk_size,
                compression=getattr(config.global_config.cache, 'compression', 'zstd'),
                verify_hashes=config.global_config.cache.verify_hashes,
                max_memory_usage=self.max_memory_usage,
                enable_memory_tracking=self.enable_memory_tracking
            )
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {str(e)}")
            return None

    @contextmanager
    def _track_memory(self, context: str):
        """Track memory usage during operations."""
        if not (self.enable_memory_tracking and torch.cuda.is_available()):
            yield
            return
            
        try:
            start_mem = torch.cuda.memory_allocated()
            yield
            end_mem = torch.cuda.memory_allocated()
            
            self.stats.memory_allocated = end_mem
            self.stats.peak_memory = max(self.stats.peak_memory, end_mem)
            
            if end_mem - start_mem > 1e8:  # 100MB
                logger.warning(f"High memory allocation in {context}: {(end_mem - start_mem) / 1e6:.1f}MB")
        except Exception as e:
            logger.error(f"Memory tracking failed in {context}: {str(e)}")
            raise

    def _process_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Process single image with memory optimization and enhanced error handling."""
        with self._track_memory("image_processing"):
            try:
                # Validate input image with detailed checks
                if not isinstance(image, Image.Image):
                    raise ProcessingError(
                        f"Expected PIL.Image, got {type(image)}",
                        {"input_type": str(type(image))}
                    )
                
                # Validate image properties
                if not hasattr(image, 'size') or not hasattr(image, 'mode'):
                    raise ProcessingError(
                        "Invalid image object - missing required attributes",
                        {"image_attrs": dir(image)}
                    )
                
                # Validate image dimensions with bounds checking
                w, h = image.size
                if w <= 0 or h <= 0 or w > 32768 or h > 32768:  # Add reasonable upper bounds
                    raise ProcessingError(
                        f"Invalid image dimensions: {image.size}",
                        {"width": w, "height": h}
                    )
                
                # Validate image mode
                if image.mode not in ('RGB', 'RGBA'):
                    raise ProcessingError(
                        f"Unsupported image mode: {image.mode}",
                        {"supported_modes": ['RGB', 'RGBA']}
                    )
                
                # Validate target size with detailed context
                tw, th = target_size
                if tw <= 0 or th <= 0 or tw > 32768 or th > 32768:
                    raise ProcessingError(
                        f"Invalid target size: {target_size}",
                        {"target_width": tw, "target_height": th}
                    )
                
                # Setup CUDA stream with better error handling
                compute_stream = None
                if device and device.type == 'cuda':
                    try:
                        compute_stream = torch.cuda.Stream()
                        if not compute_stream.query():
                            logger.warning("CUDA stream creation succeeded but stream is not ready")
                    except RuntimeError as cuda_err:
                        logger.warning(
                            f"CUDA stream creation failed: {cuda_err}. Falling back to CPU processing.",
                            extra={"cuda_device": torch.cuda.current_device()}
                        )
                
                # Process image with enhanced error tracking
                with create_stream_context(compute_stream) if compute_stream else nullcontext():
                    try:
                        # Convert to tensor with validation
                        tensor = self.transforms(image)
                        
                        # Validate tensor properties
                        if not isinstance(tensor, torch.Tensor):
                            raise ProcessingError(
                                f"Transform output is not a tensor: {type(tensor)}",
                                {"output_type": str(type(tensor))}
                            )
                        
                        if tensor.dim() != 3:  # CHW format
                            raise ProcessingError(
                                f"Expected 3D tensor (CHW), got shape {tensor.shape}",
                                {"tensor_dims": tensor.dim(), "expected_dims": 3}
                            )
                        
                        if tensor.shape[0] not in (1, 3, 4):  # Valid channel counts
                            raise ProcessingError(
                                f"Invalid number of channels: {tensor.shape[0]}",
                                {"channels": tensor.shape[0], "valid_channels": [1, 3, 4]}
                            )
                        
                        # Add batch dimension
                        tensor = tensor.unsqueeze(0)
                        
                        # Move to device with enhanced error handling
                        if device:
                            try:
                                # Check device memory before transfer
                                if device.type == 'cuda':
                                    free_mem = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                                    required_mem = tensor.element_size() * tensor.nelement()
                                    if required_mem > free_mem:
                                        torch_gc()  # Try to free memory
                                        free_mem = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                                        if required_mem > free_mem:
                                            raise ProcessingError(
                                                "Insufficient GPU memory for tensor transfer",
                                                {
                                                    "required_memory": required_mem,
                                                    "free_memory": free_mem,
                                                    "tensor_size": tensor.shape
                                                }
                                            )
                                
                                # Transfer tensor
                                tensor = tensor.to(
                                    device=device,
                                    memory_format=torch.channels_last,
                                    non_blocking=True
                                )
                                
                                # Pin memory and record stream
                                if device.type == 'cuda':
                                    pin_tensor_(tensor)
                                    if compute_stream:
                                        tensors_record_stream(compute_stream, tensor)
                                        
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    torch_gc()
                                    raise ProcessingError(
                                        "GPU out of memory during tensor transfer",
                                        {
                                            "allocated_memory": torch.cuda.memory_allocated(device),
                                            "max_memory": torch.cuda.get_device_properties(device).total_memory,
                                            "tensor_size": tensor.shape
                                        }
                                    ) from e
                                raise ProcessingError(
                                    f"Device transfer failed: {str(e)}",
                                    {"device": str(device), "tensor_size": tensor.shape}
                                ) from e
                        
                        # Remove batch dimension and ensure float type
                        tensor = tensor.squeeze(0)
                        if not tensor.is_floating_point():
                            tensor = tensor.float()
                        
                        # Validate final tensor
                        if torch.isnan(tensor).any():
                            raise ProcessingError(
                                "Output tensor contains NaN values",
                                {"tensor_stats": {
                                    "min": tensor.min().item(),
                                    "max": tensor.max().item(),
                                    "mean": tensor.mean().item()
                                }}
                            )
                        
                        self.stats.processed_images += 1
                        return tensor
                        
                    except Exception as transform_error:
                        error_context = {
                            'transform_error': str(transform_error),
                            'tensor_shape': getattr(tensor, 'shape', None),
                            'tensor_dtype': getattr(tensor, 'dtype', None),
                            'transform_pipeline': str(self.transforms),
                            'device': str(device)
                        }
                        raise ProcessingError("Transform pipeline failed", error_context) from transform_error
                    
            except ProcessingError:
                raise  # Re-raise ProcessingError with context
            except Exception as e:
                error_context = {
                    'image_size': getattr(image, 'size', None),
                    'image_mode': getattr(image, 'mode', None),
                    'image_format': getattr(image, 'format', None),
                    'target_size': target_size,
                    'device': str(device),
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                raise ProcessingError("Image processing failed", error_context) from e

    def _process_cached_item(
        self,
        index: int,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve and process cached item."""
        if not self.cache_manager:
            return None
            
        try:
            file_path = self.image_paths[index]
            cached_data = self.cache_manager.get_cached_item(file_path, device)
            
            if cached_data:
                self.stats.cache_hits += 1
                return cached_data
                
            self.stats.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error processing cached item at index {index}: {str(e)}")
            return None

    def _cache_processed_item(
        self,
        item_data: Dict[str, Any],
        image_path: Path,
        pixel_values: torch.Tensor
    ) -> None:
        """Cache processed item with memory optimization."""
        try:
            # Generate latents and embeddings
            with torch.no_grad():
                text_embeddings = self.latent_preprocessor.encode_prompt(
                    [item_data["text"]],
                    proportion_empty_prompts=getattr(self.config.data, 'proportion_empty_prompts', 0.0),
                    is_train=self.is_train
                )
                
                vae_latents = self.latent_preprocessor.encode_images(
                    pixel_values.unsqueeze(0) if pixel_values.dim() == 3 else pixel_values
                )
            
            # Prepare cache data
            latent_data = {
                "image": pixel_values,
                "vae_latents": vae_latents["model_input"]
            }
            
            metadata = {
                "original_size": item_data["original_size"],
                "crop_top_left": item_data["crop_top_left"],
                "target_size": item_data["target_size"],
                "cached_at": time.time()
            }
            
            # Save to cache
            self.cache_manager.save_preprocessed_data(
                latent_data=latent_data,
                text_embeddings=text_embeddings,
                metadata=metadata,
                file_path=image_path
            )
            
        except Exception as e:
            logger.error(f"Failed to cache processed item: {str(e)}")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with caching support."""
        try:
            # Check cache first
            cached_item = self._process_cached_item(idx)
            if cached_item:
                return {
                    "pixel_values": cached_item["latent"]["image"],
                    "text": self.captions[idx],
                    "original_size": cached_item["metadata"].get("original_size"),
                    "crop_top_left": cached_item["metadata"].get("crop_top_left", (0, 0)),
                    "target_size": cached_item["metadata"].get("target_size"),
                    "prompt_embeds": cached_item["text_embeddings"].get("prompt_embeds"),
                    "pooled_prompt_embeds": cached_item["text_embeddings"].get("pooled_prompt_embeds"),
                    "loss_weight": self.tag_weighter.get_caption_weight(self.captions[idx])
                        if self.tag_weighter else 1.0
                }
            
            # Process image if not cached
            with self._track_memory("item_processing"):
                # Load and validate image
                image_path = self._validate_image_path(self.image_paths[idx])
                image = self._load_image(image_path)
                bucket_idx = self.bucket_indices[idx]
                target_h, target_w = self.buckets[bucket_idx]
                crop_top, crop_left = self._get_crop_coordinates(image, target_h, target_w)
                
                # Process image
                image = crop(image, crop_top, crop_left, target_h, target_w)
                pixel_values = self._process_image(image, (target_w, target_h))
                
                original_size = image.size
                
                # Prepare item data
                item_data = {
                    "pixel_values": pixel_values,
                    "text": self.captions[idx],
                    "original_size": original_size,
                    "crop_top_left": (crop_top, crop_left),
                    "target_size": (target_h, target_w),
                    "loss_weight": self.tag_weighter.get_caption_weight(self.captions[idx])
                        if self.tag_weighter else 1.0
                }
                
                # Cache processed data if manager available
                if self.cache_manager and self.latent_preprocessor:
                    self._cache_processed_item(
                        item_data,
                        image_path,
                        pixel_values
                    )
                
                return item_data
                
        except Exception as e:
            logger.error(f"Error getting dataset item {idx}: {str(e)}")
            raise

    def _validate_image_path(self, path: Union[str, Path]) -> Path:
        """Validate and convert image path."""
        try:
            if isinstance(path, (list, tuple)):
                path = path[0] if path else None
            
            if not path:
                raise ValueError("Invalid image path")
                
            path = Path(convert_windows_path(path, make_absolute=True))
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
                
            return path
            
        except Exception as e:
            raise DatasetError(f"Invalid image path: {str(e)}")

    def _get_crop_coordinates(
        self,
        image: Image.Image,
        target_h: int,
        target_w: int
    ) -> Tuple[int, int]:
        """Get crop coordinates based on configuration."""
        if self.config.training.center_crop:
            crop_top = max(0, (image.height - target_h) // 2)
            crop_left = max(0, (image.width - target_w) // 2)
        else:
            crop_top = torch.randint(0, max(1, image.height - target_h), (1,)).item()
            crop_left = torch.randint(0, max(1, image.width - target_w), (1,)).item()
            
        return crop_top, crop_left

    def _load_image(self, path: Path) -> Image.Image:
        """Load image with error handling."""
        try:
            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            raise ImageLoadError(f"Failed to load image {path}: {str(e)}")

    def collate_fn(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with optimized memory handling."""
        with self._track_memory("collate"):
            try:
                # Create or reuse buffers
                if not hasattr(self, '_collate_buffers'):
                    self._collate_buffers = self._create_collate_buffers(
                        len(examples),
                        examples[0]["pixel_values"].shape
                    )
                
                # Get streams for pipelined operations
                transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                
                with create_stream_context(transfer_stream):
                    # Copy to pinned buffer
                    if 'pixel_values' in self._collate_buffers:
                        for i, example in enumerate(examples):
                            self._collate_buffers['pixel_values'][i].copy_(
                                example["pixel_values"],
                                non_blocking=True
                            )
                            
                    # Transfer to device with compute stream
                    if compute_stream and 'device_storage' in self._collate_buffers:
                        with torch.cuda.stream(compute_stream):
                            compute_stream.wait_stream(transfer_stream)
                            
                            # Move to device and optimize format
                            pixel_values = self._collate_buffers['device_storage']
                            pixel_values.copy_(
                                self._collate_buffers['pixel_values'],
                                non_blocking=True
                            )
                            
                            # Record stream and clean up
                            tensors_record_stream(compute_stream, pixel_values)
                            for buffer in self._collate_buffers.values():
                                if buffer.is_pinned():
                                    unpin_tensor_(buffer)
                    else:
                        # CPU fallback
                        pixel_values = torch.stack([
                            example["pixel_values"] for example in examples
                        ])
                
                # Create batch dictionary
                batch = {
                    "pixel_values": pixel_values,
                    "text": [example["text"] for example in examples],
                    "original_sizes": [example["original_size"] for example in examples],
                    "crop_top_lefts": [example["crop_top_left"] for example in examples],
                    "target_sizes": [example["target_size"] for example in examples],
                    "loss_weights": torch.tensor([
                        example["loss_weight"] for example in examples
                    ], dtype=torch.float32)
                }
                
                # Add latent data if available
                if "prompt_embeds" in examples[0] and "pooled_prompt_embeds" in examples[0]:
                    batch.update({
                        "prompt_embeds": torch.stack([
                            example["prompt_embeds"] for example in examples
                        ]),
                        "pooled_prompt_embeds": torch.stack([
                            example["pooled_prompt_embeds"] for example in examples
                        ])
                    })
                
                return batch
                
            except Exception as e:
                raise ProcessingError("Collate failed", {
                    'batch_size': len(examples),
                    'error': str(e)
                })
            finally:
                torch_gc()

    def _create_collate_buffers(self, batch_size: int, example_shape: torch.Size) -> Dict[str, torch.Tensor]:
        """Create reusable buffers for collate function."""
        with self._track_memory("create_buffers"):
            try:
                buffers = {}
                
                if torch.cuda.is_available():
                    # Create pinned buffers for efficient transfers
                    buffers['pixel_values'] = torch.empty(
                        (batch_size,) + example_shape,
                        pin_memory=True
                    )
                    
                    # Create device buffers for final storage
                    device = torch.device('cuda')
                    buffers['device_storage'] = torch.empty(
                        (batch_size,) + example_shape,
                        device=device,
                        memory_format=torch.channels_last
                    )
                    
                return buffers
                
            except Exception as e:
                raise ProcessingError("Failed to create collate buffers", {
                    'batch_size': batch_size,
                    'shape': example_shape,
                    'error': str(e)
                })

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Clean up preprocessing pipeline
            if self.preprocessing_pipeline:
                self.preprocessing_pipeline.__exit__(exc_type, exc_val, exc_tb)
                
            # Clean up CUDA resources
            if torch.cuda.is_available():
                if hasattr(self, 'streams'):
                    for stream in self.streams.values():
                        stream.synchronize()
                torch_gc()
                
            # Log final statistics
            if self.enable_memory_tracking:
                logger.info(
                    f"Dataset statistics:\n"
                    f"- Processed images: {self.stats.processed_images}/{self.stats.total_images}\n"
                    f"- Failed images: {self.stats.failed_images}\n"
                    f"- Cache hits: {self.stats.cache_hits}\n"
                    f"- Cache misses: {self.stats.cache_misses}\n"
                    f"- Peak memory: {self.stats.peak_memory / 1e9:.2f}GB"
                )
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    latent_preprocessor: Optional[LatentPreprocessor] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    """Create dataset instance with configuration."""
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        latent_preprocessor=latent_preprocessor,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
