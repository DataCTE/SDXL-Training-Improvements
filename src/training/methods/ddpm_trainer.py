"""DDPM trainer implementation for SDXL with extreme speedups."""
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor

from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(torch, "compile"):
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(batch, generator)
        return self._compute_loss_impl(batch, generator)

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        try:
            # Extract latents from cached format with shape logging
            if "latent" in batch:
                latents = batch["latent"].get("model_input", None)
                if latents is None:
                    latents = batch["latent"].get("latent", {}).get("model_input", None)
                if latents is None:
                    raise ValueError("Could not find model_input in latent data")
            else:
                # Try direct model_input key as fallback
                latents = batch.get("model_input")
                if latents is None:
                    raise KeyError("No latent data found in batch")

            # Process tensors with detailed shape logging
            logger.debug("=== DDPM Training Step Tensor Shapes ===")
            logger.debug(f"→ Input latents:          {latents.shape}")

            # Process noise and timesteps
            noise = torch.randn_like(latents, generator=generator)
            logger.debug(f"→ Generated noise:        {noise.shape}")
            
            timesteps = torch.randint(
                0, 
                self.noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=latents.device
            )
            logger.debug(f"→ Timesteps:             {timesteps.shape}")

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            logger.debug(f"→ Noisy latents:         {noisy_latents.shape}")

            # Extract embeddings from cached format
            if "embeddings" in batch:
                prompt_embeds = batch["embeddings"].get("prompt_embeds")
                pooled_prompt_embeds = batch["embeddings"].get("pooled_prompt_embeds")
                if prompt_embeds is None or pooled_prompt_embeds is None:
                    raise ValueError("Missing required embeddings in cached data")
            else:
                # Try direct keys as fallback
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")
                if prompt_embeds is None or pooled_prompt_embeds is None:
                    raise KeyError("No embeddings found in batch")

            logger.debug(f"→ Prompt embeds:         {prompt_embeds.shape}")
            logger.debug(f"→ Pooled prompt embeds:  {pooled_prompt_embeds.shape}")

            # Get add_time_ids with shape logging
            add_time_ids = get_add_time_ids(
                original_sizes=batch.get("original_sizes", [(1024, 1024)]),
                crop_top_lefts=batch.get("crop_top_lefts", [(0, 0)]),
                target_sizes=batch.get("target_sizes", [(1024, 1024)]),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device
            )
            logger.debug(f"→ Add time ids:          {add_time_ids.shape}")
            
            # Log UNet input shapes
            logger.debug("=== UNet Forward Pass Input Shapes ===")
            logger.debug(f"→ Noisy latents:         {noisy_latents.shape}")
            logger.debug(f"→ Timesteps:             {timesteps.shape}")
            logger.debug(f"→ Prompt embeds:         {prompt_embeds.shape}")
            logger.debug(f"→ Time ids:              {add_time_ids.shape}")
            logger.debug(f"→ Text embeds:           {pooled_prompt_embeds.shape}")
            
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "time_ids": add_time_ids, 
                    "text_embeds": pooled_prompt_embeds
                },
            ).sample
            
            logger.debug(f"→ Predicted noise:       {noise_pred.shape}")

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            logger.debug(f"→ Loss value:            {loss.item():.6f}")
            logger.debug("=====================================")
            
            return {"loss": loss}

        except Exception as e:
            logger.error("=== DDPM Error State Tensor Shapes ===")
            error_shapes = {
                "latents": getattr(latents, "shape", None) if "latents" in locals() else None,
                "noise": getattr(noise, "shape", None) if "noise" in locals() else None,
                "noisy_latents": getattr(noisy_latents, "shape", None) if "noisy_latents" in locals() else None,
                "prompt_embeds": getattr(prompt_embeds, "shape", None) if "prompt_embeds" in locals() else None,
                "pooled_prompt_embeds": getattr(pooled_prompt_embeds, "shape", None) if "pooled_prompt_embeds" in locals() else None,
                "add_time_ids": getattr(add_time_ids, "shape", None) if "add_time_ids" in locals() else None,
                "noise_pred": getattr(noise_pred, "shape", None) if "noise_pred" in locals() else None
            }
            for k, v in error_shapes.items():
                logger.error(f"→ {k}: {v}")
            logger.error("=====================================")
            raise
