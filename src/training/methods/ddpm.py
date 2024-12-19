"""DDPM training method implementation."""
from typing import Dict, Optional
import torch
import torch.nn.functional as F

from .base import TrainingMethod
from ..noise import generate_noise, get_add_time_ids

class DDPMMethod(TrainingMethod):
    """Denoising Diffusion Probabilistic Models training method."""
    
    def __init__(self, prediction_type: str = "epsilon"):
        """Initialize DDPM method.
        
        Args:
            prediction_type: Type of prediction (epsilon/v_prediction)
        """
        self.prediction_type = prediction_type
        
    @property
    def name(self) -> str:
        return "ddpm"
        
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        noise_scheduler: Optional[object] = None,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute DDPM training loss."""
        # Get batch inputs
        latents = batch["model_input"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        
        # Add noise
        noise = generate_noise(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
            layout=latents
        )
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get time embeddings
        add_time_ids = get_add_time_ids(
            batch["original_sizes"],
            batch["crop_top_lefts"],
            batch["target_sizes"],
            dtype=prompt_embeds.dtype,
            device=latents.device
        )
        
        # Predict noise
        noise_pred = model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        ).sample
        
        # Compute loss
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
            
        loss = F.mse_loss(noise_pred, target, reduction="none")
        loss = loss.mean([1, 2, 3])
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"]
            
        loss = loss.mean()
        
        return {"loss": loss}
