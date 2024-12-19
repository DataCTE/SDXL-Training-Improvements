"""Flow Matching training method implementation."""
import torch
from typing import Dict, Optional

from .base import TrainingMethod
from ..flow_matching import (
    sample_logit_normal,
    optimal_transport_path,
    compute_flow_matching_loss
)
from ..noise import get_add_time_ids

class FlowMatchingMethod(TrainingMethod):
    """Flow Matching training method."""
    
    @property
    def name(self) -> str:
        return "flow_matching"
        
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        noise_scheduler: Optional[object] = None,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute Flow Matching training loss."""
        # Get batch inputs
        x1 = batch["model_input"]
        
        # Sample time values from logit-normal
        t = sample_logit_normal(
            (x1.shape[0],),
            device=x1.device,
            dtype=x1.dtype,
            mean=0.0,
            std=1.0
        )
        
        # Sample initial points from standard normal
        x0 = torch.randn_like(x1)
        
        # Compute optimal transport path points
        xt = optimal_transport_path(x0, x1, t)
        
        # Get conditioning
        condition_embeddings = {
            "prompt_embeds": batch["prompt_embeds"],
            "added_cond_kwargs": {
                "text_embeds": batch["pooled_prompt_embeds"],
                "time_ids": get_add_time_ids(
                    batch["original_sizes"],
                    batch["crop_top_lefts"],
                    batch["target_sizes"],
                    dtype=batch["prompt_embeds"].dtype,
                    device=x1.device
                )
            }
        }
        
        # Compute loss
        loss = compute_flow_matching_loss(
            model,
            x0,
            x1,
            t,
            condition_embeddings
        )
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"].mean()
            
        return {"loss": loss}
