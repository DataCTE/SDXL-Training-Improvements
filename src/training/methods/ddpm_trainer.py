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
        try:
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
            
        except Exception as e:
            self.tensor_logger.handle_error(e, {
                'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                'has_compiled_loss': hasattr(self, '_compiled_loss')
            })
            raise

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        try:
            # Extract latents and move to correct device/dtype
            if "latent" in batch:
                self.tensor_logger.log_checkpoint("Initial Batch", {
                    "latent.model_input": batch["latent"].get("model_input"),
                    "latent.latent": batch["latent"].get("latent"),
                    "prompt_embeds": batch["embeddings"]["prompt_embeds"],
                    "pooled_prompt_embeds": batch["embeddings"]["pooled_prompt_embeds"]
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

            # Move to device and convert dtype
            target_dtype = self.unet.dtype
            latents = latents.to(device=self.unet.device, dtype=target_dtype)
            self.tensor_logger.log_checkpoint("Processed Latents", {"latents": latents})

            # Generate noise with matching dtype
            if generator is not None:
                with torch.random.fork_rng(devices=[latents.device]):
                    torch.random.manual_seed(generator.initial_seed())
                    noise = torch.randn_like(latents, dtype=target_dtype)
            else:
                noise = torch.randn_like(latents, dtype=target_dtype)
            self.tensor_logger.log_checkpoint("Generated Noise", {"noise": noise})

            # Get timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            self.tensor_logger.log_checkpoint("Timesteps", {"timesteps": timesteps})

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            self.tensor_logger.log_checkpoint("Noisy Latents", {"noisy_latents": noisy_latents})

            # Get embeddings and ensure correct dtype
            prompt_embeds = batch["embeddings"]["prompt_embeds"].to(device=self.unet.device, dtype=target_dtype)
            pooled_prompt_embeds = batch["embeddings"]["pooled_prompt_embeds"].to(device=self.unet.device, dtype=target_dtype)
            self.tensor_logger.log_checkpoint("Embeddings", {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds
            })

            # Get add_time_ids with correct dtype
            add_time_ids = get_add_time_ids(
                original_sizes=batch["original_sizes"],
                crop_top_lefts=batch["crop_top_lefts"],
                target_sizes=batch["target_sizes"],
                dtype=target_dtype,  # Use target_dtype consistently
                device=self.unet.device
            )
            self.tensor_logger.log_checkpoint("Time IDs", {"add_time_ids": add_time_ids})

            # Forward pass
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "time_ids": add_time_ids,
                    "text_embeds": pooled_prompt_embeds
                }
            ).sample
            self.tensor_logger.log_checkpoint("Model Output", {"noise_pred": noise_pred})

            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            return {"loss": loss}

        except Exception as e:
            self.tensor_logger.handle_error(e, {
                'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                'device': str(self.unet.device),
                'unet_dtype': str(self.unet.dtype),
                'error': str(e),
                'shape_history': [log for log in self.tensor_logger.get_shape_history()],
                'traceback': traceback.format_exc()
            })
            raise RuntimeError(f"Loss computation failed - see logs for detailed shape history: {str(e)}") from e
