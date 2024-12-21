"""High-performance dataset implementation for SDXL training with optimized memory handling."""
import os
import threading
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

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
from .preprocessing import LatentPreprocessor, TagWeighter, create_tag_weighter

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
        enable_memory_tracking: bool = True
    ):
        """Initialize dataset with enhanced memory management."""
        super().__init__()
        
        # Initialize statistics tracking
        self.stats = DatasetStats()
        self.enable_memory_tracking = enable_memory_tracking
        
        # Setup CUDA optimizations
        if torch.cuda.is_available():
            self._setup_cuda_optimizations()
            
        # Process and validate paths with error tracking
        self.image_paths = self._validate_paths(image_paths)
        self.stats.total_images = len(self.image_paths)
        
        # Store configuration
        self.config = config
        self.captions = captions
        self.latent_preprocessor = latent_preprocessor
        self.tag_weighter = tag_weighter or self._create_tag_weighter(config, captions)
        self.is_train = is_train
        
        # Setup image processing parameters
        self._setup_image_config()
        
        # Create and validate buckets
        self.buckets = self._create_buckets()
        self.bucket_indices = self._assign_buckets()
        
        # Setup transforms with memory optimization
        self.train_transforms = self._create_transforms()
        
        # Initialize memory tracking if enabled
        if enable_memory_tracking and torch.cuda.is_available():
            self._init_memory_tracking()

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = True

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

    @contextmanager
    def _track_memory(self, context: str):
        """Track memory usage with context manager."""
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
        """Process single image with optimized memory handling."""
        with self._track_memory("image_processing"):
            try:
                # Input validation
                if not isinstance(image, Image.Image):
                    raise ProcessingError(f"Expected PIL.Image, got {type(image)}")
                    
                # Create processing streams
                transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                
                # Resize image with bounds checking
                image = self._resize_image(image, target_size)
                
                # Apply transforms with CUDA stream optimization
                with create_stream_context(compute_stream):
                    # Convert to tensor and add batch dimension
                    tensor = self.train_transforms(image).unsqueeze(0)
                    
                    # Optimize memory format for device
                    if device and device.type == 'cuda':
                        tensor = tensor.to(
                            memory_format=torch.channels_last,
                            device=device,
                            non_blocking=True
                        )
                        
                        # Pin memory and record stream
                        pin_tensor_(tensor)
                        tensors_record_stream(compute_stream, tensor)
                        
                # Remove batch dimension
                tensor = tensor.squeeze(0)
                
                self.stats.processed_images += 1
                return tensor
                
            except Exception as e:
                error_context = {
                    'target_size': target_size,
                    'device': str(device),
                    'error': str(e)
                }
                raise ProcessingError("Image processing failed", error_context) from e

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

    def collate_fn(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Optimized collate function with efficient memory handling."""
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
                
                # Process with latent preprocessor if available
                if self.latent_preprocessor is not None:
                    return self._process_with_latents(
                        examples,
                        pixel_values,
                        compute_stream
                    )
                    
                # Return raw inputs
                return {
                    "pixel_values": pixel_values,
                    "text": [example["text"] for example in examples],
                    "original_sizes": [example["original_size"] for example in examples],
                    "crop_top_lefts": [example["crop_top_left"] for example in examples],
                    "target_sizes": [example["target_size"] for example in examples],
                    "loss_weights": torch.tensor([
                        example["loss_weight"] for example in examples
                    ], dtype=torch.float32)
                }
                
            except Exception as e:
                raise ProcessingError("Collate failed", {
                    'batch_size': len(examples),
                    'error': str(e)
                })
            finally:
                # Clean up
                torch_gc()

    def _process_with_latents(
        self,
        examples: List[Dict],
        pixel_values: torch.Tensor,
        compute_stream: Optional[torch.cuda.Stream]
    ) -> Dict[str, torch.Tensor]:
        """Process batch with latent preprocessor."""
        with self._track_memory("latent_processing"):
            try:
                with create_stream_context(compute_stream):
                    # Process text embeddings
                    text_embeddings = self.latent_preprocessor.encode_prompt(
                        [example["text"] for example in examples],
                        proportion_empty_prompts=self.config.data.proportion_empty_prompts,
                        is_train=self.is_train
                    )
                    
                    # Process VAE embeddings
                    vae_embeddings = self.latent_preprocessor.encode_images(
                        pixel_values
                    )
                    
                    return {
                        "pixel_values": pixel_values,
                        "prompt_embeds": text_embeddings["prompt_embeds"],
                        "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"],
                        "model_input": vae_embeddings["model_input"],
                        "original_sizes": [example["original_size"] for example in examples],
                        "crop_top_lefts": [example["crop_top_left"] for example in examples],
                        "target_sizes": [example["target_size"] for example in examples],
                        "loss_weights": torch.tensor([
                            example["loss_weight"] for example in examples
                        ], dtype=torch.float32)
                    }
                    
            except Exception as e:
                raise ProcessingError("Latent processing failed", {
                    'batch_size': len(examples),
                    'error': str(e)
                })
            finally:
                torch_gc()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Clean up CUDA resources
            if torch.cuda.is_available():
                for stream in self.streams.values():
                    stream.synchronize()
                torch_gc()
                
            # Log final statistics
            if self.enable_memory_tracking:
                logger.info(
                    f"Dataset statistics:\n"
                    f"- Processed images: {self.stats.processed_images}/{self.stats.total_images}\n"
                    f"- Failed images: {self.stats.failed_images}\n"
                    f"- Peak memory: {self.stats.peak_memory / 1e9:.2f}GB"
                )
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")