"""Flow Matching trainer implementation."""
import logging
from typing import Dict, Optional

import torch

from src.core.distributed import is_main_process
from src.core.logging import log_metrics
from src.training.flow_matching import (
    sample_logit_normal,
    optimal_transport_path,
    compute_flow_matching_loss
)
from src.training.noise import get_add_time_ids
from src.training.methods.base import BaseSDXLTrainer

logger = logging.getLogger(__name__)

class FlowMatchingTrainer(BaseSDXLTrainer):
    """SDXL trainer using Flow Matching method."""
    
    @property
    def method_name(self) -> str:
        return "flow_matching"

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
        t = sample_logit_normal(
            (x1.shape[0],),
            device=x1.device,
            dtype=x1.dtype,
            mean=0.0,
            std=1.0
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
        loss = compute_flow_matching_loss(
            self.unet,
            x0,
            x1,
            t,
            condition_embeddings
        )
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"].mean()
            
        return {"loss": loss}
