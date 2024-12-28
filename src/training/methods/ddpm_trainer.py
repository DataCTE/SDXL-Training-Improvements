"""DDPM trainer implementation for SDXL with extreme speedups."""
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union
from torch import Tensor

from src.core.logging import setup_logging
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = setup_logging(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing DDPMTrainer")
        # Add shape logging buffer
        self._shape_logs = []
        if hasattr(torch, "compile"):
            logger.debug("Compiling loss computation function")
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def _log_tensor_shapes(self, data: Union[torch.Tensor, Dict], path: str = "", step: str = "") -> None:
        """Store tensor shapes in the shape logging buffer."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._log_tensor_shapes(value, current_path, step)
        elif isinstance(data, torch.Tensor):
            shape_info = {
                'step': step,
                'path': path,
                'shape': tuple(data.shape),
                'dtype': str(data.dtype),
                'device': str(data.device),
                'stats': {
                    'min': float(data.min().cpu().item()),
                    'max': float(data.max().cpu().item()),
                    'mean': float(data.mean().cpu().item()),
                    'std': float(data.std().cpu().item()) if data.numel() > 1 else 0.0
                }
            }
            self._shape_logs.append(shape_info)

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
        """Compute training loss with detailed shape logging."""
        try:
            # Log initial batch shapes
            self._log_tensor_shapes(batch, step="initial_batch")

            # Extract latents with shape logging
            if "latent" in batch:
                self._log_tensor_shapes(batch["latent"], step="latent_extraction")
                if "model_input" in batch["latent"]:
                    latents = batch["latent"]["model_input"]
                elif "latent" in batch["latent"] and "model_input" in batch["latent"]["latent"]:
                    latents = batch["latent"]["latent"]["model_input"]
                else:
                    raise ValueError("Could not find model_input in latent data")
            else:
                latents = batch.get("model_input")
                if latents is None:
                    raise KeyError("No latent data found in batch")
            
            self._log_tensor_shapes(latents, "latents", "after_extraction")

            # Generate noise with shape logging
            noise = torch.randn_like(latents, generator=generator)
            self._log_tensor_shapes(noise, "noise", "after_generation")

            # Generate timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            self._log_tensor_shapes(timesteps, "timesteps", "after_generation")

            # Add noise to latents with shape logging
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            self._log_tensor_shapes(noisy_latents, "noisy_latents", "after_noise_addition")

            # Extract embeddings with shape logging
            if "embeddings" in batch:
                prompt_embeds = batch["embeddings"].get("prompt_embeds")
                pooled_prompt_embeds = batch["embeddings"].get("pooled_prompt_embeds")
            else:
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")

            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Missing required embeddings")

            self._log_tensor_shapes(prompt_embeds, "prompt_embeds", "before_unet")
            self._log_tensor_shapes(pooled_prompt_embeds, "pooled_prompt_embeds", "before_unet")

            # Get add_time_ids with shape logging
            add_time_ids = get_add_time_ids(
                original_sizes=batch.get("original_sizes", [(1024, 1024)]),
                crop_top_lefts=batch.get("crop_top_lefts", [(0, 0)]),
                target_sizes=batch.get("target_sizes", [(1024, 1024)]),
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device
            )
            self._log_tensor_shapes(add_time_ids, "add_time_ids", "before_unet")

            # UNet forward pass with shape logging
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "time_ids": add_time_ids,
                    "text_embeds": pooled_prompt_embeds
                },
            ).sample
            self._log_tensor_shapes(noise_pred, "noise_pred", "after_unet")

            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            return {"loss": loss}

        except Exception as e:
            # Create detailed error report with shape history
            error_report = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "shape_history": self._shape_logs,
                "traceback": traceback.format_exc()
            }
            
            # Log the full error report
            logger.error(
                "Loss computation failed with shape history:",
                extra={
                    "error_report": error_report,
                    "shape_logs": self._shape_logs,
                    "stack_info": True
                }
            )
            
            # Clear shape logs for next attempt
            self._shape_logs = []
            
            raise RuntimeError(f"Loss computation failed with detailed shapes logged above: {str(e)}") from e
