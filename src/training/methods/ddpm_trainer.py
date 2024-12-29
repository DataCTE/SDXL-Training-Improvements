"""DDPM trainer implementation for SDXL with config-based v-prediction option."""
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
from torch import Tensor

from src.core.logging import get_logger, TensorLogger
from src.training.methods.base import TrainingMethod
from src.data.config import Config

logger = get_logger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def parameters(self):
        return self.up_proj.parameters()

    def _prepare_time_ids(
        self,
        original_sizes,
        crop_top_lefts, 
        target_sizes,
        dtype,
        device
    ) -> torch.Tensor:
        """Prepare time embeddings for SDXL conditioning."""
        add_time_ids = [
            list(original_size) + list(crop_top_left) + list(target_size)
            for original_size, crop_top_left, target_size 
            in zip(original_sizes, crop_top_lefts, target_sizes)
        ]
        add_time_ids = torch.tensor(add_time_ids, dtype=dtype, device=device)
        return add_time_ids

    def __init__(self, unet: torch.nn.Module, config: Config):
        super().__init__(unet, config)
        # Update input dimension to match concatenated embeddings
        self.up_proj = nn.Linear(1536, 1280).to(self.unet.device, self.unet.dtype)
        self.logger.debug("Initializing DDPMTrainer")

        # Validate noise scheduler initialization
        if not hasattr(self, 'noise_scheduler'):
            raise ValueError("noise_scheduler not initialized by parent class")
        
        if self.noise_scheduler is None:
            raise ValueError("noise_scheduler is None")

        self.logger.debug(f"Noise scheduler type: {type(self.noise_scheduler)}")

        # Create dedicated tensor logger for shape/diagnostics
        self.tensor_logger = TensorLogger(self.logger)
        self._shape_logs = []

        # Read from config whether we do "epsilon" or "v_prediction"
        self.prediction_type = config.model.prediction_type  # e.g. "v_prediction" or "epsilon"

        # Compile for speed if available
        if hasattr(torch, "compile"):
            self.logger.debug("Compiling DDPM loss computation function")
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

    def _up_project_if_needed(self, embeds: torch.Tensor) -> torch.Tensor:
        # Check if embeddings need up-projection based on their last dimension
        if embeds.shape[-1] != 1280:
            embeds = self.up_proj(embeds)
        return embeds

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss with metrics logging."""
        try:
            if hasattr(self, '_compiled_loss'):
                metrics = self._compiled_loss(batch, generator)
            else:
                metrics = self._compute_loss_impl(batch, generator)

            # Log input batch shapes for debugging
            self._log_tensor_shapes(batch, step="input")

            # Update metrics logger (just storing scalar metrics for UI/tracking)
            self.metrics_logger.update({
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in metrics.items()
            })
            return metrics

        except Exception as e:
            # Error handling + shape history
            self.tensor_logger.handle_error(e, {
                'error': str(e),
                'error_type': type(e).__name__,
                'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                'traceback': traceback.format_exc()
            })
            raise RuntimeError(f"Loss computation failed - see logs for shape history: {str(e)}") from e

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """
        Core logic for DDPM loss computation.
        Branches to either epsilon-pred or v-pred based on config.
        """
        try:
            # Log initial batch state
            self.tensor_logger.log_checkpoint("Initial Batch", {
                "latent": batch.get("latent"),
                "prompt_embeds": batch.get("prompt_embeds"),
                "pooled_prompt_embeds": batch.get("pooled_prompt_embeds")
            })
            
            # Process latents
            latent_dict = batch.get("latent", None)
            if isinstance(latent_dict, dict):
                if "model_input" in latent_dict:
                    latents = latent_dict["model_input"]
                elif "latent" in latent_dict and latent_dict["latent"] is not None and isinstance(latent_dict["latent"], dict):
                    if "model_input" in latent_dict["latent"]:
                        latents = latent_dict["latent"]["model_input"]
                    else:
                        raise ValueError("Could not find 'model_input' in 'latent' dictionary")
                else:
                    raise ValueError("Could not find model_input in latent data")
            else:
                latents = batch.get("model_input")
                if latents is None:
                    raise KeyError("No latent data found in batch")

            # Move latents to correct device and dtype
            latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)
            
            # Remove extra dimensions - latents should be [batch_size, channels, height, width]
            if latents.dim() == 5:
                # If shape is [batch_size, 1, channels, height, width]
                latents = latents.squeeze(1)
            
            # Validate latent dimensions
            if latents.dim() != 4:
                raise ValueError(f"Latents must be 4D [batch_size, channels, height, width], got shape {latents.shape}")

            self.tensor_logger.log_checkpoint("Processed Latents", {"latents": latents})

            # Process embeddings once
            prompt_embeds = batch.get("prompt_embeds", None)
            pooled_prompt_embeds = batch.get("pooled_prompt_embeds", None)
            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Missing prompt_embeds or pooled_prompt_embeds in batch")

            # Move embeddings to the correct device and dtype
            prompt_embeds = prompt_embeds.to(self.unet.device, self.unet.dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.unet.device, self.unet.dtype)

            # Process time embeddings once
            original_sizes = batch.get("original_sizes")
            crop_top_lefts = batch.get("crop_top_lefts")
            target_sizes = [(self.config.model.resolution, self.config.model.resolution) 
                           for _ in range(len(original_sizes))]

            add_time_ids = self._prepare_time_ids(
                original_sizes=original_sizes,
                crop_top_lefts=crop_top_lefts,
                target_sizes=target_sizes,
                dtype=self.unet.dtype,
                device=self.unet.device
            )

            # Log shapes for debugging
            self.tensor_logger.log_checkpoint("Processed Inputs", {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "add_time_ids": add_time_ids,
                "shapes": {
                    "prompt_embeds": prompt_embeds.shape,
                    "pooled_prompt_embeds": pooled_prompt_embeds.shape,
                    "add_time_ids": add_time_ids.shape,
                    "latents": latents.shape
                }
            })

            # ----------------------------------------------------------
            # 2. Generate noise & timesteps
            # ----------------------------------------------------------
            if generator is not None:
                with torch.random.fork_rng(devices=[latents.device]):
                    torch.random.manual_seed(generator.initial_seed())
                    noise = torch.randn_like(latents, dtype=self.unet.dtype)
            else:
                noise = torch.randn_like(latents, dtype=self.unet.dtype)

            self.tensor_logger.log_checkpoint("Generated Noise", {"noise": noise})

            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            self.tensor_logger.log_checkpoint("Timesteps", {"timesteps": timesteps})
            self.tensor_logger.log_checkpoint("Noise Scheduler State", {
                "alphas_cumprod": self.noise_scheduler.alphas_cumprod,
                "timesteps": timesteps,
            })

            # ----------------------------------------------------------
            # 3. Add noise to latents (DDPM forward noising)
            # ----------------------------------------------------------
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            self.tensor_logger.log_checkpoint("Noisy Latents", {"noisy_latents": noisy_latents})

            # ----------------------------------------------------------
            # 4. Forward pass with all conditioning
            # ----------------------------------------------------------
            assert latents.dim() == 4, f"Expected latents to be 4D, got shape {latents.shape}"
            assert prompt_embeds.dim() == 3, f"Expected prompt_embeds to be 3D, got shape {prompt_embeds.shape}"

            noise_pred = self.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample

            self.tensor_logger.log_checkpoint("Model Output", {"noise_pred": noise_pred})

            # ----------------------------------------------------------
            # 6. Branch for ε-pred or v-pred
            # ----------------------------------------------------------
            if self.prediction_type == "v_prediction":
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
                sigma_t = (1.0 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
                # latents is x_0
                v_target = alpha_t * noise - sigma_t * latents
                loss = F.mse_loss(noise_pred.float(), v_target.float(), reduction="mean")
            else:
                # Default to epsilon-pred
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            return {"loss": loss}

        except Exception as e:
            self.tensor_logger.handle_error(e, {
                'error': str(e),
                'error_type': type(e).__name__,
                'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                'device': str(self.unet.device),
                'unet_dtype': str(self.unet.dtype),
                'shape_history': [log for log in self.tensor_logger.get_shape_history()],
                'traceback': traceback.format_exc()
            })
            raise RuntimeError(f"Loss computation failed - see logs for shape history: {str(e)}") from e
