"""Flow Matching trainer implementation."""
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.core.distributed import is_main_process
from src.core.logging import log_metrics
from ...training.schedulers import get_scheduler_parameters, get_sigmas, get_add_time_ids
from src.training.trainers.SDXLTrainer import BaseSDXLTrainer

logger = logging.getLogger(__name__)

class FlowMatchingTrainer(BaseSDXLTrainer):
    """SDXL trainer using Flow Matching method."""
    
    @property
    def method_name(self) -> str:
        return "flow_matching"

    def sample_logit_normal(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        mean: float = 0.0,
        std: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Sample from logit-normal distribution.
        
        Args:
            shape: Output tensor shape
            device: Target device
            dtype: Target dtype
            mean: Mean of underlying normal distribution
            std: Standard deviation of underlying normal distribution
            generator: Optional random generator
            
        Returns:
            Samples from logit-normal distribution
        """
        # Sample from normal distribution
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        
        # Transform to logit-normal
        return torch.sigmoid(normal)

    def optimal_transport_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute optimal transport path between samples.
        
        Args:
            x0: Initial samples
            x1: Target samples
            t: Time values in [0,1]
            
        Returns:
            Interpolated samples at time t
        """
        # Expand time dimension
        t = t.view(-1, 1, 1, 1)
        
        # Linear interpolation
        xt = (1 - t) * x0 + t * x1
        return xt

    def compute_velocity(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute velocity field prediction.
        
        Args:
            model: UNet model
            xt: Current samples
            t: Time values
            condition_embeddings: Conditioning information
            
        Returns:
            Predicted velocity field
        """
        # Get model prediction
        with torch.set_grad_enabled(self.training):
            v_pred = model(
                xt,
                t,
                encoder_hidden_states=condition_embeddings["prompt_embeds"],
                added_cond_kwargs=condition_embeddings["added_cond_kwargs"]
            ).sample
            
        return v_pred

    def compute_flow_matching_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        condition_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Flow Matching training loss.
        
        Args:
            model: UNet model
            x0: Initial samples
            x1: Target samples
            t: Time values
            condition_embeddings: Conditioning information
            
        Returns:
            Flow Matching loss
        """
        # Get current point on optimal transport path
        xt = self.optimal_transport_path(x0, x1, t)
        
        # Compute ground truth velocity
        # v_t = dx_t/dt = x1 - x0 for linear interpolation
        v_true = x1 - x0
        
        # Get model's velocity prediction
        v_pred = self.compute_velocity(model, xt, t, condition_embeddings)
        
        # Compute MSE loss between predicted and true velocities
        loss = F.mse_loss(v_pred, v_true, reduction="none")
        loss = loss.mean(dim=[1, 2, 3])  # Mean over CHW dimensions
        
        return loss

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute Flow Matching training loss.
        
        Args:
            batch: Training batch
            generator: Optional random generator
            
        Returns:
            Dict with loss and metrics
        """
        # Get batch inputs
        x1 = batch["model_input"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        
        # Sample time values from logit-normal
        t = self.sample_logit_normal(
            (x1.shape[0],),
            device=x1.device,
            dtype=x1.dtype,
            mean=0.0,
            std=1.0,
            generator=generator
        )
        
        # Sample initial points from standard normal
        x0 = torch.randn_like(x1)
        
        # Get conditioning embeddings
        add_time_ids = get_add_time_ids(
            batch["original_sizes"],
            batch["crop_top_lefts"],
            batch["target_sizes"],
            dtype=prompt_embeds.dtype,
            device=x1.device
        )
        
        condition_embeddings = {
            "prompt_embeds": prompt_embeds,
            "added_cond_kwargs": {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        }
        
        # Compute Flow Matching loss
        loss = self.compute_flow_matching_loss(
            self.unet,
            x0,
            x1,
            t,
            condition_embeddings
        )
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"]
            
        loss = loss.mean()
        
        return {"loss": loss}
