"""Latent preprocessing utilities for SDXL training."""
import logging
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union

from src.core.memory.tensor import tensors_to_device_, torch_gc
from src.core.types import DataType, ModelWeightDtypes
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.config import Config
from src.models.encoders.clip import encode_clip

logger = logging.getLogger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: Config,
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer,
        text_encoder_one: CLIPTextModel,
        text_encoder_two: CLIPTextModelWithProjection,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        use_cache: bool = True
    ):
        """Initialize the latent preprocessor for SDXL training.
        
        Args:
            config: Training configuration
            tokenizer_one: First CLIP tokenizer
            tokenizer_two: Second CLIP tokenizer
            text_encoder_one: First CLIP text encoder
            text_encoder_two: Second CLIP text encoder with projection
            vae: VAE model
            device: Target device
            use_cache: Whether to cache embeddings
        """
        self.config = config
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.device = torch.device(device)
        self.use_cache = use_cache and config.global_config.cache.use_cache

        # Set up cache paths if enabled
        if self.use_cache:
            self.cache_dir = Path(config.global_config.cache.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.text_cache_path = self.cache_dir / "text_embeddings.pt"
            self.vae_cache_path = self.cache_dir / "vae_latents.pt"
            
            # Clear cache if configured
            if config.global_config.cache.clear_cache_on_start:
                self.clear_cache()

    def encode_prompt(
        self,
        prompt_batch: List[str],
        proportion_empty_prompts: float = 0,
        is_train: bool = True,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts to CLIP embeddings.
        
        Args:
            prompt_batch: List of prompts to encode
            proportion_empty_prompts: Proportion of empty prompts to use
            is_train: Whether this is training data
            return_tensors: Whether to return tensors or keep on device
            
        Returns:
            Dict with prompt_embeds and pooled_prompt_embeds
        """
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

        # Tokenize prompts
        tokens_1 = self.tokenizer_one(
            captions,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_2 = self.tokenizer_two(
            captions,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode with both encoders
        with torch.no_grad():
            # First encoder
            text_encoder_1_output, _ = encode_clip(
                text_encoder=self.text_encoder_one,
                tokens=tokens_1,
                default_layer=-2
            )

            # Second encoder
            text_encoder_2_output, pooled_prompt_embeds = encode_clip(
                text_encoder=self.text_encoder_two,
                tokens=tokens_2,
                default_layer=-2,
                add_pooled_output=True
            )

            # Combine encoder outputs
            prompt_embeds = torch.concat(
                [text_encoder_1_output, text_encoder_2_output],
                dim=-1
            )

        if return_tensors:
            return {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
            }
        
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds
        }

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Encode images to VAE latents with optimized memory handling."""
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(list(pixel_values))
            
        # Optimize memory format and pin memory
        pixel_values = pixel_values.to(memory_format=torch.channels_last).float()
        if torch.cuda.is_available():
            pixel_values = pixel_values.pin_memory()
            
        latents = []
        for idx in range(0, len(pixel_values), batch_size):
            batch = pixel_values[idx:idx + batch_size]
            with torch.no_grad():
                # Use CUDA streams for pipelined compute and transfer
                if torch.cuda.is_available():
                    with torch.cuda.stream(torch.cuda.Stream()) as compute_stream:
                        batch_latents = self.vae.encode(batch).latent_dist.sample()
                        batch_latents = batch_latents * self.vae.config.scaling_factor
                        
                    with torch.cuda.stream(torch.cuda.Stream()) as transfer_stream:
                        transfer_stream.wait_stream(compute_stream)
                        latents.append(batch_latents.cpu())
                else:
                    batch_latents = self.vae.encode(batch).latent_dist.sample()
                    batch_latents = batch_latents * self.vae.config.scaling_factor
                    latents.append(batch_latents.cpu())

        latents = torch.cat(latents)
        return {"model_input": latents}

    def preprocess_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        cache: bool = True
    ) -> Dataset:
        """Preprocess and cache embeddings for a dataset."""
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
