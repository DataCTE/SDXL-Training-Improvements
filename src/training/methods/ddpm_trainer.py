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
            if isinstance(batch["model_input"], list):
                latents_list = batch["model_input"]
            else:
                latents_list = [lat for lat in batch["model_input"]]

            # Force print the original shapes
            print(f"Original shapes: {[lat.shape for lat in latents_list]}")

            # First normalize orientation - make all landscape (width > height)
            oriented_latents = []
            for idx, lat in enumerate(latents_list):
                h, w = lat.shape[-2:]
                print(f"Processing latent {idx}: h={h}, w={w}")
                
                # Always make width larger than height
                if h > w:
                    print(f"Rotating latent {idx}")
                    lat = lat.transpose(-1, -2)
                
                print(f"After orientation: {lat.shape}")
                oriented_latents.append(lat)

            # Now all tensors should be in landscape orientation
            # Get the target size from the first tensor
            target_h = min(oriented_latents[0].shape[-2], oriented_latents[0].shape[-1])
            target_w = max(oriented_latents[0].shape[-2], oriented_latents[0].shape[-1])

            print(f"Target dimensions: {target_h}x{target_w}")

            # Resize all tensors to match target size
            processed_latents = []
            for idx, lat in enumerate(oriented_latents):
                if lat.shape[-2:] != (target_h, target_w):
                    print(f"Resizing latent {idx} from {lat.shape[-2:]} to {(target_h, target_w)}")
                    lat = F.interpolate(
                        lat,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                processed_latents.append(lat)
                print(f"Final shape for latent {idx}: {lat.shape}")

            # Verify shapes before stacking
            shapes = [lat.shape for lat in processed_latents]
            print(f"Shapes before stacking: {shapes}")
            if len(set(str(s) for s in shapes)) != 1:  # Convert to string for proper comparison
                raise ValueError(f"Inconsistent tensor shapes after processing: {shapes}")

            # Stack the processed tensors
            latents = torch.stack(processed_latents, dim=0)
            print(f"Final stacked shape: {latents.shape}")
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                
            logger.debug(f"Final latents shape: {latents.shape}")

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
