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
    ) -> Dict[str, torch.Tensor]:
        try:
            print("\n=== Starting DDPM Loss Computation ===")
            print("\nInput Batch Contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                elif isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                elif isinstance(value, list):
                    print(f"{key}: list of length {len(value)}")
                    if value and isinstance(value[0], torch.Tensor):
                        print(f"  First element: shape={value[0].shape}, dtype={value[0].dtype}, device={value[0].device}")
            
            # First validate model input
            if "model_input" not in batch:
                raise KeyError("Missing model_input in batch")

            # Process latents first
            if isinstance(batch["model_input"], list):
                latents_list = batch["model_input"]
            else:
                latents_list = [lat for lat in batch["model_input"]]

            print(f"\nInitial Batch:")
            print(f"Number of latents: {len(latents_list)}")
            for idx, lat in enumerate(latents_list):
                print(f"Latent {idx}: shape={lat.shape}, dtype={lat.dtype}, device={lat.device}")

            # Process embeddings - handle both direct and cached formats
            print("\nProcessing Embeddings:")
            if "embeddings" in batch:
                # Handle cached format
                embeddings = batch["embeddings"]
                prompt_embeds = embeddings.get("prompt_embeds")
                pooled_prompt_embeds = embeddings.get("pooled_prompt_embeds")
            else:
                # Try direct format
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")

            if prompt_embeds is None or pooled_prompt_embeds is None:
                print("\nEmbeddings structure:")
                for key, value in batch.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for subkey, subval in value.items():
                            if isinstance(subval, torch.Tensor):
                                print(f"  {subkey}: shape={subval.shape}, dtype={subval.dtype}")
                            else:
                                print(f"  {subkey}: type={type(subval)}")
                    elif isinstance(value, torch.Tensor):
                        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"{key}: type={type(value)}")
                raise KeyError("Missing required embeddings")

            print(f"Found embeddings:")
            print(f"prompt_embeds shape: {prompt_embeds.shape}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}")
            print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}, dtype={pooled_prompt_embeds.dtype}, device={pooled_prompt_embeds.device}")

            # Process sizes
            original_sizes = batch.get("original_sizes", [(1024, 1024)] * len(latents_list))
            crop_top_lefts = batch.get("crop_top_lefts", [(0, 0)] * len(latents_list))
            target_sizes = batch.get("target_sizes", [(1024, 1024)] * len(latents_list))

            print("\nSize Information:")
            print(f"Original sizes: {original_sizes}")
            print(f"Crop top lefts: {crop_top_lefts}")
            print(f"Target sizes: {target_sizes}")

            # Stack latents
            print("\nStacking Latents:")
            latents = torch.stack(latents_list, dim=0) if isinstance(batch["model_input"], list) else batch["model_input"]
            print(f"Stacked latents shape: {latents.shape}, dtype={latents.dtype}, device={latents.device}")

            # Generate noise
            print("\nGenerating Noise:")
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator
            )
            print(f"Generated noise shape: {noise.shape}, dtype={noise.dtype}, device={noise.device}")

            # Get timesteps
            print("\nGenerating Timesteps:")
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            print(f"Timesteps shape: {timesteps.shape}, dtype={timesteps.dtype}, device={timesteps.device}")

            # Add noise to latents
            print("\nAdding Noise to Latents:")
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            print(f"Noisy latents shape: {noisy_latents.shape}, dtype={noisy_latents.dtype}, device={noisy_latents.device}")

            # Get time embeddings
            print("\nGenerating Time Embeddings:")
            add_time_ids = get_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"], 
                batch["target_sizes"],
                dtype=prompt_embeds.dtype,
                device=latents.device
            )
            print(f"Time embeddings shape: {add_time_ids.shape}, dtype={add_time_ids.dtype}, device={add_time_ids.device}")

            # Get model prediction
            print("\nGetting Model Prediction:")
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample
            print(f"Model prediction shape: {noise_pred.shape}, dtype={noise_pred.dtype}, device={noise_pred.device}")

            # Compute loss
            print("\nComputing Loss:")
            if self.config.training.prediction_type == "epsilon":
                print("Using epsilon prediction type")
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                print("Using v_prediction type")
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.config.training.prediction_type}")

            print(f"Target shape: {target.shape}, dtype={target.dtype}, device={target.device}")

            loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3])
            print(f"Initial loss shape: {loss.shape}, dtype={loss.dtype}, device={loss.device}")

            if "loss_weights" in batch:
                print("Applying loss weights")
                loss = loss * batch["loss_weights"]
                print(f"Weighted loss shape: {loss.shape}")

            loss = loss.mean()
            print(f"Final loss value: {loss.item()}")

            torch_sync()
            return {"loss": loss}

        except Exception as e:
            print(f"\nERROR in tensor processing: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise
