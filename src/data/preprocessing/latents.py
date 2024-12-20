"""Latent preprocessing utilities for SDXL training."""
from pathlib import Path
from src.core.logging.logging import setup_logging
import torch
from typing import Dict, List, Optional, Union

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.config import Config
from src.models.encoders.clip import encode_clip

logger = setup_logging(__name__, level="INFO")

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
            from src.utils.paths import convert_windows_path
            self.cache_dir = Path(convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for individual files
            self.text_cache_dir = Path(convert_windows_path(self.cache_dir / "text", make_absolute=True))
            self.image_cache_dir = Path(convert_windows_path(self.cache_dir / "image", make_absolute=True))
            self.text_cache_dir.mkdir(parents=True, exist_ok=True)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)
            
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
        cache: bool = True,
        compression: Optional[str] = "zstd"
    ) -> Dataset:
        """Preprocess and cache embeddings for a dataset.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for processing
            cache: Whether to use disk caching
            compression: Compression format ("zstd", "gzip", or None)
            
        Returns:
            Dataset with added embeddings
        """
        # Try loading individual cache files first
        if cache and self.use_cache:
            try:
                all_prompt_embeds = []
                all_pooled_embeds = []
                all_vae_latents = []
                cache_hit = True

                for idx in range(len(dataset)):
                    text_cache = self.text_cache_dir / f"{idx}_text.pt"
                    vae_cache = self.image_cache_dir / f"{idx}_vae.pt"
                    
                    if text_cache.exists() and vae_cache.exists():
                        text_data = torch.load(text_cache)
                        vae_data = torch.load(vae_cache)
                        
                        if "prompt_embeds" in text_data and "pooled_prompt_embeds" in text_data:
                            all_prompt_embeds.append(text_data["prompt_embeds"])
                            all_pooled_embeds.append(text_data["pooled_prompt_embeds"])
                            all_vae_latents.append(vae_data["model_input"])
                        else:
                            cache_hit = False
                            break
                    else:
                        cache_hit = False
                        break

                if cache_hit:
                    logger.info("Successfully loaded individual cached embeddings")
                    dataset = dataset.add_column("prompt_embeds", torch.stack(all_prompt_embeds))
                    dataset = dataset.add_column("pooled_prompt_embeds", torch.stack(all_pooled_embeds))
                    dataset = dataset.add_column("model_input", torch.stack(all_vae_latents))
                    return dataset
                else:
                    logger.info("Some cache files missing, recomputing all embeddings")
                    
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}, recomputing...")

        # Process text embeddings
        logger.info("Computing text embeddings...")
        text_embeddings = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            try:
                batch = dataset[idx:idx + batch_size]
                batch_texts = []
                
                # Safely extract text from batch
                for text_item in batch["text"]:
                    if isinstance(text_item, (list, tuple)):
                        # Take first non-empty caption if multiple
                        valid_captions = [c for c in text_item if c and isinstance(c, str)]
                        batch_texts.append(valid_captions[0] if valid_captions else "")
                    elif isinstance(text_item, str):
                        batch_texts.append(text_item)
                    else:
                        logger.warning(f"Unexpected text type: {type(text_item)}, using empty string")
                        batch_texts.append("")
                
                if not any(batch_texts):  # Skip if all empty
                    logger.warning(f"Skipping batch {idx}: all empty captions")
                    continue
                    
                embeddings = self.encode_prompt(
                    batch_texts,
                    proportion_empty_prompts=self.config.data.proportion_empty_prompts
                )
                if embeddings is not None and all(t is not None for t in embeddings.values()):
                    text_embeddings.append(embeddings)
                else:
                    logger.warning(f"Skipping batch {idx}: invalid embeddings generated")
            except Exception as e:
                logger.error(f"Error processing text batch {idx}: {str(e)}")
                continue

        # Combine text embedding batches with validation
        if not text_embeddings:
            raise RuntimeError("No valid text embeddings were generated. Check input captions and logs for details.")
            
        try:
            # Validate embeddings before concatenation
            valid_embeds = [e for e in text_embeddings if e is not None 
                          and "prompt_embeds" in e 
                          and "pooled_prompt_embeds" in e
                          and e["prompt_embeds"] is not None
                          and e["pooled_prompt_embeds"] is not None]
            
            if not valid_embeds:
                raise RuntimeError("No valid embeddings found after filtering")
                
            text_embeddings = {
                "prompt_embeds": torch.cat([e["prompt_embeds"] for e in valid_embeds]),
                "pooled_prompt_embeds": torch.cat([e["pooled_prompt_embeds"] for e in valid_embeds])
            }
            
            logger.info(f"Successfully generated embeddings for {len(valid_embeds)} batches")
            
        except Exception as e:
            raise RuntimeError(f"Failed to concatenate text embeddings: {str(e)}")

        # Process VAE latents
        logger.info("Computing VAE latents...")
        vae_latents = []
        for idx in tqdm(range(0, len(dataset), batch_size)):
            try:
                batch = dataset[idx:idx + batch_size]
                latents = self.encode_images(batch["pixel_values"], batch_size=batch_size)
                vae_latents.append(latents)
            except Exception as e:
                logger.error(f"Error processing VAE batch {idx}: {str(e)}")
                continue

        vae_latents = {
            "model_input": torch.cat([l["model_input"] for l in vae_latents])
        }

        # Cache individual results with compression
        if cache and self.use_cache:
            logger.info("Caching individual embeddings to disk...")
            try:
                for idx in range(len(dataset)):
                    text_cache = self.text_cache_dir / f"{idx}_text.pt"
                    vae_cache = self.image_cache_dir / f"{idx}_vae.pt"
                    
                    # Prepare individual embeddings
                    text_data = {
                        "prompt_embeds": text_embeddings["prompt_embeds"][idx],
                        "pooled_prompt_embeds": text_embeddings["pooled_prompt_embeds"][idx]
                    }
                    vae_data = {
                        "model_input": vae_latents["model_input"][idx]
                    }
                    
                    # Save with optional compression
                    if compression == "zstd":
                        torch.save(text_data, text_cache, _use_new_zipfile_serialization=True)
                        torch.save(vae_data, vae_cache, _use_new_zipfile_serialization=True)
                    else:
                        torch.save(text_data, text_cache)
                        torch.save(vae_data, vae_cache)
                        
                logger.info(f"Successfully cached individual embeddings to {self.cache_dir}")
                
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
                # Continue even if caching fails

        # Add processed embeddings to dataset
        dataset = dataset.add_column("prompt_embeds", text_embeddings["prompt_embeds"])
        dataset = dataset.add_column("pooled_prompt_embeds", text_embeddings["pooled_prompt_embeds"])
        dataset = dataset.add_column("model_input", vae_latents["model_input"])

        return dataset

    def clear_cache(self):
        """Clear all embedding caches."""
        import shutil
        if self.text_cache_dir.exists():
            shutil.rmtree(self.text_cache_dir)
            self.text_cache_dir.mkdir(parents=True)
        if self.image_cache_dir.exists():
            shutil.rmtree(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True)
