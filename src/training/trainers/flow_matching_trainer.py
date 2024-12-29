"""Flow Matching trainer implementation with extreme speedups."""
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from torch import Tensor

from src.core.logging import get_logger, MetricsLogger
from src.training.trainers.base import BaseTrainer
from src.training.schedulers import get_add_time_ids
from src.data.config import Config
from src.core.distributed import is_main_process

logger = get_logger(__name__)

class FlowMatchingTrainer(BaseTrainer):
    """Flow Matching trainer with memory optimizations."""
    
    name = "flow_matching"

    def __init__(self, unet: torch.nn.Module, config: Config):
        super().__init__(unet, config)
        self.logger.debug("Initializing Flow Matching Trainer")
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(window_size=100)
        
        # Initialize tensor shape tracking
        self._shape_logs = []
        
        # Try to compile loss computation if available
        if hasattr(torch, "compile"):
            self.logger.debug("Attempting to compile loss computation")
            try:
                self._compiled_loss = torch.compile(
                    self._compute_loss_impl,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                self.logger.debug("Loss computation successfully compiled")
            except Exception as e:
                self.logger.warning(
                    "Failed to compile loss computation",
                    exc_info=True,
                    extra={'error': str(e)}
                )

    def compute_loss(self, model, batch, generator=None) -> Dict[str, Tensor]:
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
        """Compute Flow Matching loss with new data format."""
        try:
            # Get latents from batch
            latents = batch["latents"]
            
            # Apply tag weights if available
            if "tag_weights" in batch:
                tag_weights = batch["tag_weights"].to(latents.device)
                latents = latents * tag_weights.view(-1, 1, 1, 1)

            # Extract embeddings
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]

            # Use latents as target (x1)
            x1 = latents
            
            # Sample time steps
            t = self.sample_logit_normal(
                (x1.shape[0],),
                x1.device,
                x1.dtype,
                generator=generator
            )
            
            # Generate random starting point
            x0 = torch.randn_like(x1)
            
            # Get time embeddings from metadata
            add_time_ids = get_add_time_ids(
                original_sizes=batch["metadata"]["original_sizes"],
                crop_top_lefts=batch["metadata"]["crop_top_lefts"],
                target_sizes=batch["metadata"]["target_sizes"],
                dtype=prompt_embeds.dtype,
                device=x1.device
            )
            
            # Prepare conditioning embeddings
            cond_emb = {
                "prompt_embeds": prompt_embeds,
                "added_cond_kwargs": {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            }
            
            # Log shapes for debugging
            if is_main_process():
                self._log_tensor_shapes({
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }, step="input")

            # Compute flow matching loss
            loss = self.compute_flow_matching_loss(self.unet, x0, x1, t, cond_emb)
            
            # Apply any additional weights
            if "loss_weights" in batch:
                loss = loss * batch["loss_weights"].to(loss.device)
            loss = loss.mean()
            
            # Compute additional metrics
            metrics = {
                "loss": loss.detach().item(),
                "x0_norm": x0.norm().item(),
                "x1_norm": x1.norm().item(),
                "time_mean": t.mean().item(),
                "time_std": t.std().item()
            }
            
            # Update metrics logger
            self.metrics_logger.update(metrics)
            
            return {
                "loss": loss,
                "metrics": metrics
            }

        except Exception as e:
            self.logger.error(
                "Error computing Flow Matching loss",
                exc_info=True,
                extra={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'shape_logs': self._shape_logs,
                    'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                    'device_info': {
                        'x1_device': x1.device if 'x1' in locals() else None,
                        'prompt_device': prompt_embeds.device if 'prompt_embeds' in locals() else None
                    }
                }
            )
            raise

    def _log_tensor_shapes(self, tensors: Dict[str, Tensor], step: str):
        """Log tensor shapes for debugging."""
        shapes = {
            name: tensor.shape if hasattr(tensor, 'shape') else None 
            for name, tensor in tensors.items()
        }
        self._shape_logs.append({
            'step': step,
            'shapes': shapes,
            'devices': {
                name: str(tensor.device) if hasattr(tensor, 'device') else None
                for name, tensor in tensors.items()
            }
        })

    def sample_logit_normal(
        self, 
        shape, 
        device, 
        dtype, 
        mean=0.0, 
        std=1.0, 
        generator=None
    ) -> Tensor:
        """Sample from logit-normal distribution."""
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        return torch.sigmoid(normal)

    def optimal_transport_path(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Compute optimal transport path."""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_velocity(
        self, 
        model: torch.nn.Module, 
        xt: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute velocity field."""
        return model(
            xt,
            t,
            encoder_hidden_states=cond_emb["prompt_embeds"],
            added_cond_kwargs=cond_emb["added_cond_kwargs"]
        ).sample

    def compute_flow_matching_loss(
        self, 
        model: torch.nn.Module, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute flow matching loss."""
        xt = self.optimal_transport_path(x0, x1, t)
        v_true = x1 - x0
        v_pred = self.compute_velocity(model, xt, t, cond_emb)
        return F.mse_loss(v_pred, v_true, reduction="none").mean([1, 2, 3])
