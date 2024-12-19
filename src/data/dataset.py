"""Dataset implementation for SDXL training."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
from PIL import Image
from src.utils.paths import convert_windows_path
from PIL.Image import BILINEAR, FLIP_LEFT_RIGHT
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

# Local imports
from src.core.memory.tensor import (
    create_stream_context,
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_,
    torch_gc,
    tensors_to_device_,
    device_equals
)
from src.core.memory.optimizations import (
    setup_memory_optimizations,
    verify_memory_optimizations
)
from .config import Config
from .preprocessing import LatentPreprocessor, TagWeighter, create_tag_weighter

logger = logging.getLogger(__name__)

class SDXLDataset(Dataset):
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        latent_preprocessor: Optional[LatentPreprocessor] = None,
        tag_weighter: Optional[TagWeighter] = None,
        is_train: bool = True
    ):
        """Initialize SDXL dataset with memory optimizations.
        
        Args:
            config: Configuration object
            image_paths: List of paths to images
            captions: List of captions/prompts
            latent_preprocessor: Optional latent preprocessor
            tag_weighter: Optional tag weighter
            is_train: Whether this is training data
        """
        # Setup memory optimizations first
        if torch.cuda.is_available():
            setup_memory_optimizations(
                model=None,
                config=config,
                device=torch.device("cuda"),
                batch_size=config.training.batch_size
            )
        # Convert Windows paths if needed
        image_paths = [str(convert_windows_path(p, make_absolute=True)) for p in image_paths]
        # Verify all paths exist after conversion
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Path does not exist after conversion: {path}")
        """SDXL Dataset with bucketing and aspect ratio preservation.
        
        Args:
            config: Configuration object
            image_paths: List of paths to images
            captions: List of captions/prompts
            latent_preprocessor: Optional latent preprocessor for caching
            tag_weighter: Optional tag weighter for loss weighting
            is_train: Whether this is training data
        """
        self.config = config
        self.image_paths = image_paths
        self.captions = captions
        self.latent_preprocessor = latent_preprocessor
        self.tag_weighter = tag_weighter
        self.is_train = is_train
        
        # Image settings from config
        self.target_size = config.global_config.image.target_size
        self.max_size = config.global_config.image.max_size
        self.min_size = config.global_config.image.min_size
        self.bucket_step = config.global_config.image.bucket_step
        self.max_aspect_ratio = config.global_config.image.max_aspect_ratio
        
        # Create buckets based on aspect ratios
        self.buckets = self._create_buckets()
        self.bucket_indices = self._assign_buckets()
        
        # Set up transforms
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Initialize tag weighter if enabled but not provided
        if self.tag_weighter is None and config.tag_weighting.enable_tag_weighting:
            self.tag_weighter = create_tag_weighter(config, captions)
            
        # Verify memory optimizations
        if torch.cuda.is_available():
            verify_memory_optimizations(
                model=None,
                config=config,
                device=torch.device("cuda"),
                logger=logger
            )

    def _create_buckets(self) -> List[Tuple[int, int]]:
        """Get supported SDXL dimensions as buckets."""
        return self.config.global_config.image.supported_dims

    def _assign_buckets(self) -> List[int]:
        """Assign each image to closest supported SDXL dimension."""
        bucket_indices = []
        
        for img_path in self.image_paths:
            img = Image.open(img_path)
            w, h = img.size
            aspect_ratio = w / h
            
            # Find closest supported dimension
            min_diff = float('inf')
            best_bucket_idx = 0
            
            for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
                # Compare aspect ratios and total area difference
                bucket_ratio = bucket_w / bucket_h
                ratio_diff = abs(aspect_ratio - bucket_ratio)
                area_diff = abs((w * h) - (bucket_w * bucket_h))
                
                # Weighted combination of ratio and area differences
                total_diff = ratio_diff + area_diff / (1536 * 1536)  # Normalize area diff
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_bucket_idx = idx
            
            bucket_indices.append(best_bucket_idx)
            
        return bucket_indices

    def _process_image(self, image: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
        """Process image with resizing and augmentations."""
        # Resize
        image = image.resize(target_size, BILINEAR)
        
        # Random flip in training
        if self.is_train and self.config.training.random_flip and torch.rand(1).item() < 0.5:
            image = image.transpose(FLIP_LEFT_RIGHT)
            
        # Convert to tensor and normalize efficiently
        with create_stream_context(torch.cuda.current_stream()):
            image = self.train_transforms(image)
            if self.is_train and image.device.type == 'cuda':
                tensors_record_stream(torch.cuda.current_stream(), image)
        
        return image

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Tuple[int, int], float]]:
        """Get dataset item with bucketing.
        
        Returns dict with:
            - pixel_values: Processed image tensor
            - text: Caption/prompt
            - original_size: Original image dimensions
            - crop_top_left: Crop coordinates
            - target_size: Target size after bucketing
            - loss_weight: Optional tag-based loss weight
        """
        # Load and process image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Get bucket dimensions
        bucket_idx = self.bucket_indices[idx]
        target_h, target_w = self.buckets[bucket_idx]
        
        # Calculate crop coordinates
        if self.config.training.center_crop:
            crop_top = max(0, (image.height - target_h) // 2)
            crop_left = max(0, (image.width - target_w) // 2)
        else:
            crop_top = torch.randint(0, max(1, image.height - target_h), (1,)).item()
            crop_left = torch.randint(0, max(1, image.width - target_w), (1,)).item()
            
        # Crop and process image
        image = crop(image, crop_top, crop_left, target_h, target_w)
        pixel_values = self._process_image(image, (target_w, target_h))
        
        # Get caption and compute loss weight if tag weighter is enabled
        caption = self.captions[idx]
        loss_weight = (
            self.tag_weighter.get_caption_weight(caption)
            if self.tag_weighter is not None
            else 1.0
        )
        
        return {
            "pixel_values": pixel_values,
            "text": caption,
            "original_size": original_size,
            "crop_top_left": (crop_top, crop_left),
            "target_size": (target_h, target_w),
            "loss_weight": loss_weight
        }

    def collate_fn(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function with optimized memory handling and CUDA streams.
        
        Uses:
        - Multiple CUDA streams for pipelined transfers
        - Pinned memory for faster host-device transfers
        - Channels-last memory format for better throughput
        - Bucketing for efficient batch construction
        """
        # Create streams for pipelined operations
        transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Initialize variables
        pixel_values = None
        pinned_buffer = None
        
        # Pre-allocate pinned memory buffer if using CUDA
        if torch.cuda.is_available():
            buffer_shape = (len(examples),) + examples[0]["pixel_values"].shape
            pinned_buffer = torch.empty(buffer_shape, pin_memory=True)
            
        # Stack tensors with optimized memory handling
        with create_stream_context(transfer_stream):
            # Skip if no pinned buffer
            if pinned_buffer is not None:
                # Copy to pinned buffer
                for i, example in enumerate(examples):
                    pinned_buffer[i].copy_(example["pixel_values"], non_blocking=True)
            
            # Move to GPU with channels-last optimization if using CUDA
            if pinned_buffer is not None:
                pixel_values = pinned_buffer.to(
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    memory_format=torch.channels_last,
                    non_blocking=True
                )
                
                if torch.cuda.is_available():
                    tensors_record_stream(transfer_stream, pixel_values)
            else:
                # Fallback for CPU-only
                pixel_values = torch.stack([example["pixel_values"] for example in examples])
                
            loss_weights = torch.tensor([example["loss_weight"] for example in examples], dtype=torch.float32)
        
        # If using latent preprocessor, get cached embeddings
        if self.latent_preprocessor is not None:
            # Use compute stream for preprocessing
            with create_stream_context(compute_stream):
                if compute_stream is not None:
                    compute_stream.wait_stream(transfer_stream)
                    
                text_embeddings = self.latent_preprocessor.encode_prompt(
                    [example["text"] for example in examples],
                    proportion_empty_prompts=self.config.data.proportion_empty_prompts,
                    is_train=self.is_train
                )
                vae_embeddings = self.latent_preprocessor.encode_images(pixel_values)
            
            return {
                "pixel_values": pixel_values,
                "prompt_embeds": text_embeddings["prompt_embeds"],
                "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"],
                "model_input": vae_embeddings["model_input"],
                "original_sizes": [example["original_size"] for example in examples],
                "crop_top_lefts": [example["crop_top_left"] for example in examples],
                "target_sizes": [example["target_size"] for example in examples],
                "loss_weights": loss_weights
            }
        
        # Without preprocessor, return raw inputs
        return {
            "pixel_values": pixel_values,
            "text": [example["text"] for example in examples],
            "original_sizes": [example["original_size"] for example in examples],
            "crop_top_lefts": [example["crop_top_left"] for example in examples],
            "target_sizes": [example["target_size"] for example in examples],
            "loss_weights": loss_weights
        }


def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    latent_preprocessor: Optional[LatentPreprocessor] = None,
    tag_weighter: Optional[TagWeighter] = None,
    is_train: bool = True
) -> SDXLDataset:
    """Create SDXL dataset with validation.
    
    Args:
        config: Configuration object
        image_paths: List of paths to images
        captions: List of captions/prompts
        latent_preprocessor: Optional latent preprocessor
        tag_weighter: Optional tag weighter
        is_train: Whether this is training data
        
    Returns:
        Configured dataset
    """
    # Validate inputs
    assert len(image_paths) == len(captions), "Number of images and captions must match"
    assert all(Path(p).exists() for p in image_paths), "All image paths must exist"
    
    # Create dataset
    dataset = SDXLDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        latent_preprocessor=latent_preprocessor,
        tag_weighter=tag_weighter,
        is_train=is_train
    )
    
    logger.info(f"Created dataset with {len(dataset)} examples and {len(dataset.buckets)} buckets")
    return dataset
