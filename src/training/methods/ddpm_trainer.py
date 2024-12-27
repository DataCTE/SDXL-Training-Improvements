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

            # Log original shapes
            logger.info(f"Processing batch with {len(latents_list)} latents")
            for idx, lat in enumerate(latents_list):
                logger.info(f"Original latent {idx}: shape={lat.shape}")

            # Determine target orientation based on majority
            orientations = []
            for lat in latents_list:
                h, w = lat.shape[-2:]
                orientations.append('landscape' if w >= h else 'portrait')
            
            target_orientation = max(set(orientations), key=orientations.count)
            logger.info(f"Target orientation: {target_orientation}")

            # Normalize orientations
            oriented_latents = []
            for idx, lat in enumerate(latents_list):
                h, w = lat.shape[-2:]
                current_orientation = 'landscape' if w >= h else 'portrait'
                
                if current_orientation != target_orientation:
                    logger.info(f"Rotating latent {idx} from {current_orientation} to {target_orientation}")
                    lat = lat.transpose(-1, -2)
                
                oriented_latents.append(lat)
                logger.info(f"Latent {idx} after orientation: shape={lat.shape}")

            # Get target dimensions based on the majority orientation
            target_h = min(oriented_latents[0].shape[-2:])
            target_w = max(oriented_latents[0].shape[-2:])
            logger.info(f"Target dimensions: {target_h}x{target_w}")

            # Resize all tensors to match target size
            processed_latents = []
            for idx, lat in enumerate(oriented_latents):
                if lat.shape[-2:] != (target_h, target_w):
                    logger.info(f"Resizing latent {idx} from {lat.shape[-2:]} to {(target_h, target_w)}")
                    lat = F.interpolate(
                        lat,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                processed_latents.append(lat)

            # Final verification
            shapes = [lat.shape for lat in processed_latents]
            if len(set(str(s) for s in shapes)) != 1:
                shape_details = [f"latent_{i}: {s}" for i, s in enumerate(shapes)]
                raise ValueError(f"Inconsistent shapes after processing:\n" + "\n".join(shape_details))

            # Stack the processed tensors
            latents = torch.stack(processed_latents, dim=0)
            logger.info(f"Successfully stacked latents to shape: {latents.shape}")
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
