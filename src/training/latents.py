"""Latent preprocessing utilities for SDXL training."""
import logging
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..config import Config
from ..training.noise import get_add_time_ids

logger = logging.getLogger(__name__)

class LatentPreprocessor:
    def __init__(
        self,
        config: "Config",  # type: ignore
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer,
        text_encoder_one: CLIPTextModel,
        text_encoder_two: CLIPTextModel,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda"
    ):
        """Initialize the latent preprocessor for SDXL training."""
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
        """Encode text prompts to CLIP embeddings."""
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
        """Encode images to VAE latents."""
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
