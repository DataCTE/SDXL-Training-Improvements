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
            print("\n=== Starting Tensor Processing ===")  # Clear visual separator
            
            # Initial batch inspection
            if isinstance(batch["model_input"], list):
                latents_list = batch["model_input"]
            else:
                latents_list = [lat for lat in batch["model_input"]]

            print(f"\nInitial Batch:")
            print(f"Number of latents: {len(latents_list)}")
            for idx, lat in enumerate(latents_list):
                print(f"Latent {idx}: shape={lat.shape}, dtype={lat.dtype}, device={lat.device}")

            # Orientation analysis
            print("\nAnalyzing Orientations:")
            orientations = []
            for idx, lat in enumerate(latents_list):
                h, w = lat.shape[-2:]
                current_orientation = 'landscape' if w >= h else 'portrait'
                orientations.append(current_orientation)
                print(f"Latent {idx}: {h}x{w} -> {current_orientation}")
            
            target_orientation = max(set(orientations), key=orientations.count)
            print(f"\nTarget orientation: {target_orientation}")

            # Normalize orientations
            print("\nNormalizing Orientations:")
            oriented_latents = []
            for idx, lat in enumerate(latents_list):
                h, w = lat.shape[-2:]
                current_orientation = 'landscape' if w >= h else 'portrait'
                
                if current_orientation != target_orientation:
                    print(f"Rotating latent {idx} from {current_orientation} to {target_orientation}")
                    print(f"Before rotation: {lat.shape}")
                    lat = lat.transpose(-1, -2)
                    print(f"After rotation: {lat.shape}")
                else:
                    print(f"Latent {idx} already in {target_orientation} orientation: {lat.shape}")
                
                oriented_latents.append(lat)

            # Target size determination
            print("\nDetermining Target Size:")
            shapes = [lat.shape[-2:] for lat in oriented_latents]
            print(f"Available shapes: {shapes}")
            target_h = min(min(shape[-2] for shape in shapes), min(shape[-1] for shape in shapes))
            target_w = max(max(shape[-2] for shape in shapes), max(shape[-1] for shape in shapes))
            print(f"Target dimensions: {target_h}x{target_w}")

            # Resize tensors
            print("\nResizing Tensors:")
            processed_latents = []
            for idx, lat in enumerate(oriented_latents):
                if lat.shape[-2:] != (target_h, target_w):
                    print(f"Resizing latent {idx}: {lat.shape[-2:]} -> ({target_h}, {target_w})")
                    lat = F.interpolate(
                        lat,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                processed_latents.append(lat)
                print(f"Processed latent {idx} final shape: {lat.shape}")

            # Final verification
            print("\nFinal Verification:")
            shapes = [lat.shape for lat in processed_latents]
            print(f"Final shapes: {shapes}")
            if len(set(str(s) for s in shapes)) != 1:
                shape_details = [f"latent_{i}: {s}" for i, s in enumerate(shapes)]
                error_msg = "Inconsistent shapes after processing:\n" + "\n".join(shape_details)
                print(f"ERROR: {error_msg}")
                raise ValueError(error_msg)

            # Stack tensors
            print("\nStacking Tensors:")
            latents = torch.stack(processed_latents, dim=0)
            print(f"Final stacked shape: {latents.shape}")
            print("\n=== Tensor Processing Complete ===\n")

            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]

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
            print(f"\nERROR in tensor processing: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise
