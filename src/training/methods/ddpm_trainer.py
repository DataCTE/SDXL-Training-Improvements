"""DDPM trainer implementation for SDXL with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
from src.training.methods.base import make_picklable
from src.core.history import TorchHistory
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor


from src.core.memory import torch_sync, create_stream_context
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        super().__init__(*args, **kwargs)
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
        if hasattr(torch, "compile"):
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the history attribute from pickling
        state.pop('history', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the history instance and re-add the hook
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
    @make_picklable
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(model, batch, generator)
        return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        try:
            # Direct access since data is already validated
            latents = batch["model_input"]
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]

            # Get dimensions from first item and ensure landscape orientation
            h0, w0 = latents[0].shape[-2:]
            target_height = min(h0, w0)
            target_width = max(h0, w0)
            
            # Resize and rotate latents to match target dimensions
            resized_latents = []
            for lat in latents:
                h, w = lat.shape[-2:]
                # Check if we need to rotate
                needs_rotation = (h > w) != (target_height > target_width)
                
                if needs_rotation:
                    # Transpose the last two dimensions
                    lat = lat.transpose(-1, -2)
                
                # Resize if dimensions don't match
                if lat.shape[-2:] != (target_height, target_width):
                    lat = F.interpolate(
                        lat,
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False
                    )
                resized_latents.append(lat)
            
            latents = torch.cat(resized_latents, dim=0)

            # Get noise and timesteps
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

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get time embeddings
            add_time_ids = get_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"], 
                batch["target_sizes"],
                dtype=prompt_embeds.dtype,
                device=latents.device
            )

            # Get model prediction
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample

            # Compute loss
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
            logger.error(f"Error computing DDPM loss: {str(e)}", exc_info=True)
            torch_sync()
            raise
