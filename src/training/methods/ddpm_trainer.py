"""DDPM trainer implementation for SDXL."""
import logging
from typing import Dict, Optional

import torch
from src.core.memory import torch_gc, create_stream_context
from torch import Tensor
import torch.nn.functional as F

from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class DDPMTrainer(TrainingMethod):
    """SDXL trainer using DDPM method with v-prediction."""
    
    name = "ddpm"  # Class attribute for registration

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor], 
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute DDPM training loss.
        
        Args:
            batch: Training batch containing:
                - model_input: Latent tensors
                - prompt_embeds: Text embeddings
                - pooled_prompt_embeds: Pooled text embeddings
                - original_sizes: Original image sizes
                - crop_top_lefts: Crop coordinates
                - target_sizes: Target sizes
                - loss_weights: Optional per-sample weights
            generator: Optional random number generator
            
        Returns:
            Dict containing:
                - loss: Training loss tensor
        """
        try:
            # Get batch inputs
            latents = batch["model_input"]
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            
            logger.debug(f"Input shapes - latents: {latents.shape}, "
                        f"prompts: {prompt_embeds.shape}, "
                        f"pooled: {pooled_prompt_embeds.shape}")
            
            # Add noise using scheduler
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator
            )
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
            # Get time embeddings
            add_time_ids = get_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"],
                batch["target_sizes"],
                dtype=prompt_embeds.dtype,
                device=latents.device
            )
            
            # Predict noise
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample
            
            # Compute loss based on prediction type
            if self.config.training.prediction_type == "epsilon":
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.config.training.prediction_type}")
                
            loss = F.mse_loss(noise_pred, target, reduction="none")
            loss = loss.mean([1, 2, 3])
            
            # Apply loss weights if provided
            if "loss_weights" in batch:
                loss = loss * batch["loss_weights"]
                
            loss = loss.mean()
            
            # Clean up intermediate tensors  
            torch_gc()
            
            return {"loss": loss}
            
        except Exception as e:
            logger.error(f"Error computing DDPM loss: {str(e)}")
            torch_gc()  # Clean up on error
            raise
