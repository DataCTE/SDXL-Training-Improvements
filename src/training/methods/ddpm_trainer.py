"""DDPM trainer implementation for SDXL with config-based v-prediction option."""
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from torch import Tensor

from src.core.logging import get_logger, TensorLogger
from src.training.methods.base import TrainingMethod
from src.data.config import Config

logger = get_logger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def _prepare_time_ids(self, add_time_ids: torch.Tensor) -> torch.Tensor:
        """Reshape time IDs to match expected dimensions: (batch_size, 6, 1, 1)."""
        if add_time_ids.ndim == 2:
            add_time_ids = add_time_ids.unsqueeze(-1).unsqueeze(-1)

        if hasattr(self.unet, 'dtype'):
            add_time_ids = add_time_ids.to(dtype=self.unet.dtype)

        return add_time_ids

    def __init__(self, unet: torch.nn.Module, config: Config):
        super().__init__(unet, config)
        self.logger.debug("Initializing DDPMTrainer")

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

            # Update metrics logger
            self.metrics_logger.update({
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in metrics.items()
            })
            return metrics

        except Exception as e:
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
            # ----------------------------------------------------------
            # 1. Extract latents from the batch
            # ----------------------------------------------------------
            if "latent" in batch:
                self.tensor_logger.log_checkpoint("Initial Batch", {
                    "latent": batch["latent"],
                    "prompt_embeds": batch.get("prompt_embeds"),
                    "pooled_prompt_embeds": batch.get("pooled_prompt_embeds")
                })

                if "model_input" in batch["latent"]:
                    latents = batch["latent"]["model_input"]
                elif "latent" in batch["latent"] and "model_input" in batch["latent"]["latent"]:
                    latents = batch["latent"]["latent"]["model_input"]
                else:
                    raise ValueError("Could not find model_input in latent data")

                self.tensor_logger.log_checkpoint("Initial Latent Data", {
                    "latent.model_input": latents,
                    "latent.latent": batch["latent"].get("latent")
                })
            else:
                latents = batch.get("model_input")
                if latents is None:
                    raise KeyError("No latent data found in batch")

            # Move latents to device
            target_dtype = self.unet.dtype
            latents = latents.to(device=self.unet.device, dtype=target_dtype)
            self.tensor_logger.log_checkpoint("Processed Latents", {"latents": latents})

            # ----------------------------------------------------------
            # 2. Generate noise & timesteps
            # ----------------------------------------------------------
            if generator is not None:
                with torch.random.fork_rng(devices=[latents.device]):
                    torch.random.manual_seed(generator.initial_seed())
                    noise = torch.randn_like(latents, dtype=target_dtype)
            else:
                noise = torch.randn_like(latents, dtype=target_dtype)

            self.tensor_logger.log_checkpoint("Generated Noise", {"noise": noise})

            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            self.tensor_logger.log_checkpoint("Timesteps", {"timesteps": timesteps})

            # ----------------------------------------------------------
            # 3. Add noise to latents (DDPM forward noising)
            # ----------------------------------------------------------
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            self.tensor_logger.log_checkpoint("Noisy Latents", {"noisy_latents": noisy_latents})

            # ----------------------------------------------------------
            # 4. Retrieve text embeddings for SDXL
            # ----------------------------------------------------------
            # For an SDXL UNet, we need to pass prompt_embeds as `encoder_hidden_states`.
            # Otherwise, it will complain about missing the argument.
            if "prompt_embeds" not in batch or not isinstance(batch["prompt_embeds"], torch.Tensor):
                raise ValueError(
                    "SDXL UNet requires `prompt_embeds` in the batch to feed as `encoder_hidden_states`."
                )
            prompt_embeds = batch["prompt_embeds"].to(device=self.unet.device, dtype=target_dtype)

            # ----------------------------------------------------------
            # 5. UNet forward pass with conditioning
            # ----------------------------------------------------------
            noise_pred = self.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds
            ).sample

            self.tensor_logger.log_checkpoint("Model Output", {"noise_pred": noise_pred})

            # ----------------------------------------------------------
            # 6. Branch for ε-pred or v-pred
            # ----------------------------------------------------------
            if self.prediction_type == "v_prediction":
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
                sigma_t = (1.0 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
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
