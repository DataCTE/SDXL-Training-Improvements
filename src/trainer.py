"""NovelAI Diffusion V3 Trainer with optimized memory usage."""
import functools
import gc
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageOps
from PIL.Image import BILINEAR, FLIP_LEFT_RIGHT
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from training.config import Config
from training.memory import (
    configure_model_memory_format,
    setup_memory_optimizations,
    verify_memory_optimizations
)
from .training.noise import generate_noise, get_add_time_ids
from training.metrics import log_metrics as utils_log_metrics


logger = logging.getLogger(__name__)


class TagWeighter:
    def __init__(
        self,
        config: Config,
        cache_path: Optional[Path] = None
    ):
        """Initialize tag weighting system.
        
        Args:
            config: Configuration object
            cache_path: Optional path to cache tag statistics
        """
        self.config = config
        self.cache_path = cache_path or Path(config.global_config.cache.cache_dir) / "tag_weights.json"
        
        # Tag weighting settings
        self.default_weight = config.tag_weighting.default_weight
        self.min_weight = config.tag_weighting.min_weight
        self.max_weight = config.tag_weighting.max_weight
        self.smoothing_factor = config.tag_weighting.smoothing_factor
        
        # Initialize tag statistics
        self.tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_weights = defaultdict(lambda: defaultdict(lambda: self.default_weight))
        self.total_samples = 0
        
        # Tag type categories
        self.tag_types = {
            "subject": ["person", "character", "object", "animal", "vehicle", "location"],
            "style": ["art_style", "artist", "medium", "genre"],
            "quality": ["quality", "rating", "aesthetic"],
            "technical": ["camera", "lighting", "composition", "color"],
            "meta": ["source", "meta", "misc"]
        }
        
        # Load cached weights if available
        if config.tag_weighting.use_cache and self.cache_path.exists():
            self._load_cache()

    def _extract_tags(self, caption: str) -> Dict[str, List[str]]:
        """Extract tags from caption and categorize them.
        
        Args:
            caption: Input caption/prompt
            
        Returns:
            Dictionary mapping tag types to lists of tags
        """
        # Split caption into individual tags
        tags = [t.strip() for t in re.split(r'[,\(\)]', caption) if t.strip()]
        
        # Categorize tags
        categorized_tags = defaultdict(list)
        
        for tag in tags:
            tag_type = self._determine_tag_type(tag)
            categorized_tags[tag_type].append(tag)
            
        return dict(categorized_tags)

    def _determine_tag_type(self, tag: str) -> str:
        """Determine the type of a tag based on keywords and patterns.
        
        Args:
            tag: Input tag
            
        Returns:
            Tag type category
        """
        tag = tag.lower()
        
        # Check each category's keywords
        for type_name, keywords in self.tag_types.items():
            if any(keyword in tag for keyword in keywords):
                return type_name
                
        # Default to subject if no clear category
        return "subject"

    def update_statistics(self, captions: List[str]):
        """Update tag statistics from a batch of captions.
        
        Args:
            captions: List of caption strings
        """
        for caption in captions:
            self.total_samples += 1
            categorized_tags = self._extract_tags(caption)
            
            # Update counts for each tag type
            for tag_type, tags in categorized_tags.items():
                for tag in tags:
                    self.tag_counts[tag_type][tag] += 1

        # Recompute weights
        self._compute_weights()
        
        # Cache updated statistics
        if self.config.tag_weighting.use_cache:
            self._save_cache()

    def _compute_weights(self):
        """Compute tag weights based on frequency statistics."""
        for tag_type in self.tag_counts:
            type_counts = self.tag_counts[tag_type]
            total_type_count = sum(type_counts.values())
            
            if total_type_count == 0:
                continue
                
            # Compute frequency-based weights
            for tag, count in type_counts.items():
                frequency = count / self.total_samples
                
                # Apply smoothing and inverse frequency weighting
                weight = 1.0 / (frequency + self.smoothing_factor)
                
                # Normalize to configured range
                weight = np.clip(weight, self.min_weight, self.max_weight)
                self.tag_weights[tag_type][tag] = float(weight)

    def get_caption_weight(self, caption: str) -> float:
        """Get the weight for a caption based on its tags.
        
        Args:
            caption: Input caption string
            
        Returns:
            Combined weight for the caption
        """
        categorized_tags = self._extract_tags(caption)
        weights = []
        
        for tag_type, tags in categorized_tags.items():
            type_weights = [self.tag_weights[tag_type][tag] for tag in tags]
            if type_weights:
                # Use mean weight for each tag type
                weights.append(np.mean(type_weights))
        
        if not weights:
            return self.default_weight
            
        # Combine weights across types (geometric mean)
        return float(np.exp(np.mean(np.log(weights))))

    def get_batch_weights(self, captions: List[str]) -> torch.Tensor:
        """Get weights for a batch of captions.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Tensor of weights for each caption
        """
        weights = [self.get_caption_weight(caption) for caption in captions]
        return torch.tensor(weights, dtype=torch.float32)

    def _save_cache(self):
        """Save tag statistics to cache."""
        cache_data = {
            "tag_counts": {k: dict(v) for k, v in self.tag_counts.items()},
            "tag_weights": {k: dict(v) for k, v in self.tag_weights.items()},
            "total_samples": self.total_samples
        }
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f)

    def _load_cache(self):
        """Load tag statistics from cache."""
        with open(self.cache_path, 'r') as f:
            cache_data = json.load(f)
            
        self.tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_weights = defaultdict(lambda: defaultdict(lambda: self.default_weight))
        
        for tag_type, counts in cache_data["tag_counts"].items():
            self.tag_counts[tag_type].update(counts)
            
        for tag_type, weights in cache_data["tag_weights"].items():
            self.tag_weights[tag_type].update(weights)
            
        self.total_samples = cache_data["total_samples"]

    def get_tag_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics about tags and their weights.
        
        Returns:
            Dictionary with tag statistics
        """
        stats = {}
        
        for tag_type in self.tag_counts:
            type_stats = {
                "total_tags": len(self.tag_counts[tag_type]),
                "total_occurrences": sum(self.tag_counts[tag_type].values()),
                "most_common": sorted(
                    self.tag_counts[tag_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                "weight_range": (
                    min(self.tag_weights[tag_type].values()),
                    max(self.tag_weights[tag_type].values())
                )
            }
            stats[tag_type] = type_stats
            
        return stats


def create_tag_weighter(
    config: Config,
    captions: List[str],
    cache_path: Optional[Path] = None
) -> TagWeighter:
    """Create and initialize a tag weighter with a dataset.
    
    Args:
        config: Configuration object
        captions: List of all captions to initialize statistics
        cache_path: Optional path to cache
        
    Returns:
        Initialized TagWeighter
    """
    weighter = TagWeighter(config, cache_path)
    
    if not (config.tag_weighting.use_cache and weighter.cache_path.exists()):
        logger.info("Computing tag statistics...")
        weighter.update_statistics(captions)
        
    stats = weighter.get_tag_statistics()
    logger.info("Tag statistics:")
    for tag_type, type_stats in stats.items():
        logger.info(f"\n{tag_type}:")
        logger.info(f"Total unique tags: {type_stats['total_tags']}")
        logger.info(f"Total occurrences: {type_stats['total_occurrences']}")
        logger.info(f"Weight range: {type_stats['weight_range']}")
        
    return weighter




class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer,
        text_encoder_one: CLIPTextModel,
        text_encoder_two: CLIPTextModel,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda"
    ):
        """Initialize the latent preprocessor for SDXL training.
        
        Args:
            config: Configuration object
            tokenizer_one: First CLIP tokenizer
            tokenizer_two: Second CLIP tokenizer  
            text_encoder_one: First CLIP text encoder
            text_encoder_two: Second CLIP text encoder
            vae: VAE model
            device: Device to process on
        """
        self.config = config
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.device = device

        # Set up cache paths
        self.cache_dir = Path(config.global_config.cache.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_cache_path = self.cache_dir / "text_embeddings.pt"
        self.vae_cache_path = self.cache_dir / "vae_latents.pt"

    def encode_prompt(
        self,
        prompt_batch: List[str],
        proportion_empty_prompts: float = 0,
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts to CLIP embeddings.
        
        Args:
            prompt_batch: List of text prompts
            proportion_empty_prompts: Proportion of prompts to make empty
            is_train: Whether this is for training
            
        Returns:
            Dictionary with prompt_embeds and pooled_prompt_embeds
        """
        prompt_embeds_list = []

        # Process prompts
        captions = []
        for caption in prompt_batch:
            if torch.rand(1).item() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            else:
                # Take random caption if multiple provided during training
                captions.append(torch.randint(0, len(caption), (1,)).item() if is_train else caption[0])

        with torch.no_grad():
            for tokenizer, text_encoder in [(self.tokenizer_one, self.text_encoder_one), 
                                         (self.tokenizer_two, self.text_encoder_two)]:
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(self.device)
                
                prompt_embeds = text_encoder(
                    text_input_ids,
                    output_hidden_states=True,
                    return_dict=False,
                )
                
                # Get pooled and hidden state embeddings
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-1][-2]
                
                # Reshape prompt embeddings
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        # Concatenate embeddings from both text encoders
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

        return {
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
        }

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Encode images to VAE latents.
        
        Args:
            pixel_values: Tensor of pixel values [B, C, H, W]
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with model_input latents
        """
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(list(pixel_values))
            
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(self.device)

        latents = []
        for idx in range(0, len(pixel_values), batch_size):
            batch = pixel_values[idx:idx + batch_size]
            with torch.no_grad():
                batch_latents = self.vae.encode(batch).latent_dist.sample()
                batch_latents = batch_latents * self.vae.config.scaling_factor
                latents.append(batch_latents.cpu())

        latents = torch.cat(latents)
        return {"model_input": latents}

    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        cache: bool = True
    ) -> Dataset:
        """Preprocess and cache embeddings for a dataset.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
            cache: Whether to cache the results
            
        Returns:
            Preprocessed dataset with embeddings
        """
        if cache and self.text_cache_path.exists() and self.vae_cache_path.exists():
            logger.info("Loading cached embeddings...")
            text_embeddings = torch.load(self.text_cache_path)
            vae_latents = torch.load(self.vae_cache_path)
            
            dataset = dataset.add_column("prompt_embeds", text_embeddings["prompt_embeds"])
            dataset = dataset.add_column("pooled_prompt_embeds", text_embeddings["pooled_prompt_embeds"]) 
            dataset = dataset.add_column("model_input", vae_latents["model_input"])
            
            return dataset

        logger.info("Computing text embeddings...")
        text_embeddings = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[idx:idx + batch_size]
            embeddings = self.encode_prompt(
                batch["text"],
                proportion_empty_prompts=self.config.data.proportion_empty_prompts
            )
            text_embeddings.append(embeddings)

        # Combine batches
        text_embeddings = {
            "prompt_embeds": torch.cat([e["prompt_embeds"] for e in text_embeddings]),
            "pooled_prompt_embeds": torch.cat([e["pooled_prompt_embeds"] for e in text_embeddings])
        }

        logger.info("Computing VAE latents...")
        vae_latents = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[idx:idx + batch_size]
            latents = self.encode_images(batch["pixel_values"], batch_size=batch_size)
            vae_latents.append(latents)

        vae_latents = {
            "model_input": torch.cat([l["model_input"] for l in vae_latents])
        }

        # Cache results
        if cache:
            logger.info("Caching embeddings...")
            torch.save(text_embeddings, self.text_cache_path)
            torch.save(vae_latents, self.vae_cache_path)

        # Add to dataset
        dataset = dataset.add_column("prompt_embeds", text_embeddings["prompt_embeds"])
        dataset = dataset.add_column("pooled_prompt_embeds", text_embeddings["pooled_prompt_embeds"])
        dataset = dataset.add_column("model_input", vae_latents["model_input"])

        return dataset

    def clear_cache(self):
        """Clear the embedding caches."""
        if self.text_cache_path.exists():
            self.text_cache_path.unlink()
        if self.vae_cache_path.exists():
            self.vae_cache_path.unlink()

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

    def _create_buckets(self) -> List[Tuple[int, int]]:
        """Create resolution buckets based on config settings."""
        buckets = []
        min_dim = min(self.min_size)
        max_dim = max(self.max_size)
        
        # Generate dimensions
        for h in range(min_dim, max_dim + 1, self.bucket_step):
            for w in range(min_dim, max_dim + 1, self.bucket_step):
                # Check if dimensions are valid
                if h * w <= self.config.global_config.image.max_dim:
                    aspect_ratio = w / h
                    if 1/self.max_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                        buckets.append((h, w))
        
        return sorted(buckets)

    def _assign_buckets(self) -> List[int]:
        """Assign each image to closest bucket based on aspect ratio."""
        bucket_indices = []
        
        for img_path in self.image_paths:
            img = Image.open(img_path)
            w, h = img.size
            aspect_ratio = w / h
            
            # Find closest bucket
            min_diff = float('inf')
            best_bucket_idx = 0
            
            for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
                bucket_ratio = bucket_w / bucket_h
                diff = abs(aspect_ratio - bucket_ratio)
                
                if diff < min_diff:
                    min_diff = diff
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
            
        # Convert to tensor and normalize
        image = self.train_transforms(image)
        
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
        """Custom collate function for batching.
        
        Handles different image sizes within batch by using bucketing.
        """
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        loss_weights = torch.tensor([example["loss_weight"] for example in examples], dtype=torch.float32)
        
        # If using latent preprocessor, get cached embeddings
        if self.latent_preprocessor is not None:
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
    """Create SDXL dataset with validation."""
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

from training.scheduler import (
    get_karras_scalings,
    get_sigmas,
    get_scheduler_parameters,
    configure_noise_scheduler
)

class NovelAIDiffusionV3Trainer(torch.nn.Module):
    """Trainer for NovelAI Diffusion V3 with optimized memory usage."""
    
    def __init__(
        self,
        config: Config,
        model: Optional[UNet2DConditionModel] = None,
        dataset: Optional[SDXLDataset] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer with optimized memory usage."""
        super().__init__()
        
        # Store config
        self.config = config
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and move to device
        self.model = model
        if self.model is None:
            self.model = UNet2DConditionModel.from_pretrained(
                self.config.model.pretrained_model_name,
                subfolder="unet"
            )
        self.model.to(self.device)
        
        # Configure noise scheduler and get parameters
        try:
            scheduler_params = configure_noise_scheduler(self.config, self.device)
            self.scheduler = scheduler_params['scheduler']
            self.sigmas = scheduler_params['sigmas']
            self.snr_weights = scheduler_params['snr_weights']
            self.c_skip = scheduler_params['c_skip']
            self.c_out = scheduler_params['c_out']
            self.c_in = scheduler_params['c_in']
            self.num_train_timesteps = self.config.model.num_timesteps
            logger.info("Successfully configured noise scheduler")
        except Exception as e:
            logger.error(f"Failed to configure noise scheduler: {str(e)}")
            raise
        
        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        
        # Get model dtype for consistent tensor operations
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Configure model format and optimizations
        if not self._setup_model_optimizations():
            logger.warning("Training will proceed without memory optimizations")
            
        # Setup dataset and dataloader
        self.dataset = dataset
        if self.dataset is not None:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.training.batch_size,
                collate_fn=self.dataset.collate_fn,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                persistent_workers=config.data.persistent_workers,
                shuffle=True
            )
            
            # Calculate max_train_steps based on dataset size
            steps_per_epoch = len(self.dataset) // self.config.training.batch_size
            self.max_train_steps = steps_per_epoch * self.config.training.num_epochs
            logger.info(f"Training will run for {self.max_train_steps} steps")
        else:
            self.dataloader = None
            self.max_train_steps = None
            logger.warning("No dataset provided, trainer initialized without data")

    def _setup_model_optimizations(self):
        """Setup model memory optimizations."""
        try:
            # Configure memory format
            configure_model_memory_format(
                model=self.model,
                config=self.config
            )
            
            # Setup memory optimizations
            batch_size = self.config.training.batch_size
            micro_batch_size = batch_size // self.config.training.gradient_accumulation_steps
            
            memory_setup = setup_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size
            )
            
            # Verify memory setup was successful
            if not memory_setup:
                logger.warning("Memory optimizations setup failed")
                return False
            
            # Verify optimizations
            optimization_states = verify_memory_optimizations(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=logger
            )
            
            if not all(optimization_states.values()):
                logger.warning("Some memory optimizations failed to initialize")
                return False
                
            logger.info("Memory optimizations setup completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error setting up memory optimizations: {e}")
            raise

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        try:
            # Create optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.optimizer_betas,
                weight_decay=self.config.training.weight_decay,
                eps=self.config.training.optimizer_eps
            )
            
            # Calculate max_train_steps if not set
            if self.config.training.max_train_steps is None:
                logger.info("max_train_steps not set, will be calculated when dataloader is provided")
                self.max_train_steps = None
            else:
                self.max_train_steps = self.config.training.max_train_steps
            
            # Create scheduler if enabled
            if self.config.training.lr_scheduler != "none":
                if self.max_train_steps is None:
                    logger.warning("Cannot create scheduler yet as max_train_steps is not set")
                    self.lr_scheduler = None
                    return
                
                if self.config.training.lr_scheduler == "cosine":
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.max_train_steps - self.config.training.warmup_steps
                    )
                elif self.config.training.lr_scheduler == "linear":
                    self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1.0,
                        end_factor=0.1,
                        total_iters=self.max_train_steps - self.config.training.warmup_steps
                    )
            else:
                self.lr_scheduler = None
                
        except Exception as e:
            logger.error(f"Error setting up optimizer: {e}")
            raise

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted loss with SNR scaling and tag weights."""
        # Basic MSE in float32 for stability
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        
        # Average over non-batch dimensions
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        # Apply SNR weights if enabled
        if self.config.training.snr_gamma is not None and self.snr_weights is not None:
            snr = self.snr_weights[timesteps]
            loss = loss * snr
        
        # Apply tag-based sample weights if provided
        if weights is not None:
            loss = loss * weights.view(-1)
            
        return loss.mean()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute training step with memory optimization."""
        try:
            # Track memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated() / 1024**2
            
            # Get model inputs and embeddings
            if "model_input" in batch:
                # Use preprocessed VAE latents if available
                model_input = batch["model_input"].to(self.device, dtype=self.model_dtype, non_blocking=True)
            else:
                # Use raw pixel values and encode with VAE
                pixel_values = batch["pixel_values"].to(self.device, dtype=self.model_dtype, non_blocking=True)
                model_input = self.vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * self.vae.config.scaling_factor
            
            # Get text embeddings
            if "prompt_embeds" in batch:
                # Use preprocessed text embeddings
                prompt_embeds = batch["prompt_embeds"].to(self.device, non_blocking=True)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, non_blocking=True)
            else:
                # Encode text if not preprocessed (implementation needed)
                raise NotImplementedError("Raw text processing not implemented, use preprocessed embeddings")
            
            # Get SDXL conditioning
            add_time_ids = get_add_time_ids(
                original_sizes=batch["original_sizes"],
                crops_coords_top_lefts=batch["crop_top_lefts"],
                target_sizes=batch["target_sizes"],
                dtype=self.model_dtype,
                device=self.device
            )
            
            # Get loss weights from tag weighting
            loss_weights = batch.get("loss_weights", None)
            if loss_weights is not None:
                loss_weights = loss_weights.to(self.device, non_blocking=True)
            
            # Sample timesteps
            timesteps = torch.randint(0, self.num_train_timesteps, (model_input.shape[0],), device=self.device)
            
            # Generate noise
            noise = generate_noise(
                model_input.shape,
                self.device,
                self.model_dtype,
                model_input  # Use model_input as layout template
            )
            
            # Get Karras noise schedule scalings
            c_skip, c_out, c_in = get_karras_scalings(self.sigmas, timesteps)
            c_skip = c_skip.to(dtype=self.model_dtype)
            c_out = c_out.to(dtype=self.model_dtype)
            c_in = c_in.to(dtype=self.model_dtype)
            
            # Get sigmas and add noise
            sigmas = self.sigmas[timesteps].to(dtype=self.model_dtype)
            noisy_latents = model_input + sigmas[:, None, None, None] * noise
            
            # Scale input based on prediction type
            if self.config.training.prediction_type == "v_prediction":
                scaled_input = c_in[:, None, None, None] * noisy_latents
            else:
                scaled_input = noisy_latents
                
            # Forward pass
            model_pred = self.model(
                scaled_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
                return_dict=False
            )[0]
            
            # Get target based on prediction type
            if self.config.training.prediction_type == "epsilon":
                target = noise
            else:  # v_prediction
                target = (
                    c_skip[:, None, None, None] * model_input +
                    c_out[:, None, None, None] * noise
                )
            
            # Compute loss with tag weights
            loss = self.compute_loss(
                model_pred=model_pred,
                target=target,
                timesteps=timesteps,
                weights=loss_weights
            )
            
            # Log memory usage
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                mem_diff = peak_mem - start_mem
                logger.debug(f"Memory usage: {peak_mem:.0f}MB (+{mem_diff:.0f}MB)")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        if self.dataloader is None:
            raise ValueError("Dataloader not set. Call set_dataloader() first.")
        
        for batch_idx, batch in enumerate(self.dataloader):
            try:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Accumulate gradients
                for i in range(self.config.training.gradient_accumulation_steps):
                    loss = self.training_step(batch)
                    total_loss += loss.item()
                
                # Update weights
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update progress
                num_batches += 1
                self.global_step += 1
                
                # Log metrics
                if self.global_step % self.config.training.log_steps == 0:
                    self.log_metrics({
                        'loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.global_step
                    })
                
                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(self.config.paths.checkpoints_dir, epoch)
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Log epoch metrics
        self.log_metrics({
            'epoch': epoch,
            'epoch_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }, step_type="epoch")

    def save_checkpoint(self, checkpoints_dir: str, epoch: int) -> None:
        """Save model checkpoint."""
        try:
            checkpoint_path = os.path.join(
                checkpoints_dir,
                f"checkpoint_epoch_{epoch:03d}_step_{self.global_step:06d}.pt"
            )
            
            # Save checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'epoch': epoch
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, Any], step_type: str = "step") -> None:
        """Log metrics using utility function."""
        utils_log_metrics(
            metrics=metrics,
            step=self.global_step,
            is_main_process=True,  # TODO: Add distributed training support
            use_wandb=self.config.training.use_wandb,
            step_type=step_type
        )

    def set_dataloader(self, dataloader: DataLoader) -> None:
        """Set the dataloader."""
        self.dataloader = dataloader

    def __del__(self):
        """Remove wandb cleanup since it's handled by cleanup_logging"""
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass is same as training step for this trainer."""
        return self.training_step(batch)


