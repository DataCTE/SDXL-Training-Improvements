"""DDPM trainer implementation for SDXL with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor

# Force maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from src.core.memory import torch_sync, create_stream_context
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    # Compile if available
    if hasattr(torch, "compile"):
        def _compiled_loss(self, model, batch, generator=None):
            return self._compute_loss_impl(model, batch, generator)
        compute_loss = torch.compile(_compiled_loss, mode="reduce-overhead", fullgraph=True)
    else:
        compute_loss = None

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        if self.compute_loss:
            # If compiled is available, call the compiled method
            return self.compute_loss(model, batch, generator)
        else:
            # Fallback
            return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        try:
            latents = batch["model_input"]
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
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
            add_time_ids = get_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"],
                batch["target_sizes"],
                dtype=prompt_embeds.dtype,
                device=latents.device
            )
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample
            if self.config.training.prediction_type == "epsilon":
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.config.training.prediction_type}")

            loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3])
            if "loss_weights" in batch:
                loss = loss * batch["loss_weights"]
            loss = loss.mean()
            torch_sync()
            return {"loss": loss}
        except Exception as e:
            logger.error(f"Error computing DDPM loss: {str(e)}")
            torch_sync()
            raise
