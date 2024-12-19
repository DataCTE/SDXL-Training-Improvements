"""DDPM-specific trainer implementation."""
import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.core.distributed import is_main_process
from src.core.logging import log_metrics
from src.training.schedulers import get_scheduler_parameters, get_sigmas, get_add_time_ids
from src.training.trainers.SDXLTrainer import BaseSDXLTrainer
from src.training.schedulers.noise_scheduler import get_karras_sigmas
logger = logging.getLogger(__name__)

class DDPMTrainer(BaseSDXLTrainer):
    """SDXL trainer using DDPM method with v-prediction."""
    
    @property
    def method_name(self) -> str:
        return "ddpm"

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute DDPM training loss.
        
        Args:
            batch: Training batch
            generator: Optional random generator
            
        Returns:
            Dict with loss and metrics
        """
        # Get batch inputs
        latents = batch["model_input"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        
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
        
        return {"loss": loss}
