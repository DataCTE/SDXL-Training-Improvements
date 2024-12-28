"""DDPM trainer implementation for SDXL with extreme speedups."""
import traceback
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union
from torch import Tensor

from src.core.logging import get_logger, TensorLogger
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids
from src.data.config import Config

logger = get_logger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, unet: torch.nn.Module, config: Config):
        super().__init__(unet, config)
        logger = get_logger(__name__)
        # Create dedicated tensor logger for the trainer
        self.tensor_logger = TensorLogger(logger)
        
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
        if hasattr(self, '_compiled_loss'):
            metrics = self._compiled_loss(batch, generator)
        else:
            metrics = self._compute_loss_impl(batch, generator)
            
        # Log tensor shapes for debugging
        self._log_tensor_shapes(batch, step="input")
        
        # Update metrics logger
        self.metrics_logger.update(
            {k: v.item() if isinstance(v, torch.Tensor) else v 
             for k, v in metrics.items()}
        )
        
        return metrics

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss with detailed shape logging."""
        try:
            # Log initial batch shapes
            self.tensor_logger.log_checkpoint("Initial Batch", batch)

            # Extract latents with shape logging
            if "latent" in batch:
                self.tensor_logger.log_checkpoint("Latent Extraction", {"latent": batch["latent"]})
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
            
            self.tensor_logger.log_checkpoint("Processed Latents", {"latents": latents})

            # Generate noise with shape logging
            noise = torch.randn_like(latents, generator=generator)
            self.tensor_logger.log_checkpoint("Generated Noise", {"noise": noise})

            # Generate timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            self.tensor_logger.log_checkpoint("Timesteps", {"timesteps": timesteps})

            # Add noise to latents with shape logging
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            self.tensor_logger.log_checkpoint("Noisy Latents", {"noisy_latents": noisy_latents})

            # Extract embeddings with shape logging
            if "embeddings" in batch:
                prompt_embeds = batch["embeddings"].get("prompt_embeds")
                pooled_prompt_embeds = batch["embeddings"].get("pooled_prompt_embeds")
            else:
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")

            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Missing required embeddings")

            self.tensor_logger.log_checkpoint("Embeddings", {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds
            })

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
            
            error_context = {
                'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                'device': str(batch['model_input'].device) if isinstance(batch, dict) and 'model_input' in batch else 'unknown',
                'step': self.global_step if hasattr(self, 'global_step') else None,
                'error_report': error_report
            }
            # Use tensor logger's error handling
            self.tensor_logger.handle_error(e, error_context)
            
            # Clear shape logs for next attempt
            self._shape_logs = []
            
            raise RuntimeError(f"Loss computation failed - see logs for detailed shape history: {str(e)}") from e
