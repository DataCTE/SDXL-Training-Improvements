"""Flow Matching trainer implementation with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor

from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class FlowMatchingTrainer(TrainingMethod):
    name = "flow_matching"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(torch, "compile"):
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
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
        try:
            # Process latents
            latents_list = (batch["model_input"] 
                          if isinstance(batch["model_input"], list) 
                          else [lat for lat in batch["model_input"]])

            # Extract embeddings
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]

            # Stack latents if needed
            x1 = (torch.stack(latents_list, dim=0) 
                  if isinstance(batch["model_input"], list) 
                  else batch["model_input"])
            
            # Sample time steps
            t = self.sample_logit_normal(
                (x1.shape[0],),
                x1.device,
                x1.dtype,
                generator=generator
            )
            
            # Generate random starting point
            x0 = torch.randn_like(x1)
            
            # Get time embeddings
            add_time_ids = get_add_time_ids(
                batch["original_sizes"],
                batch["crop_top_lefts"],
                batch["target_sizes"],
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
            
            # Compute flow matching loss
            loss = self.compute_flow_matching_loss(self.unet, x0, x1, t, cond_emb)
            
            if "loss_weights" in batch:
                loss = loss * batch["loss_weights"]
            loss = loss.mean()
            
            return {"loss": loss}

        except Exception as e:
            logger.error(f"Error computing Flow Matching loss: {str(e)}", exc_info=True)
            raise

    def sample_logit_normal(self, shape, device, dtype, mean=0.0, std=1.0, generator=None):
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        return torch.sigmoid(normal)

    def optimal_transport_path(self, x0, x1, t):
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_velocity(self, model, xt, t, cond_emb):
        return model(
            xt,
            t,
            encoder_hidden_states=cond_emb["prompt_embeds"],
            added_cond_kwargs=cond_emb["added_cond_kwargs"]
        ).sample

    def compute_flow_matching_loss(self, model, x0, x1, t, cond_emb):
        xt = self.optimal_transport_path(x0, x1, t)
        v_true = x1 - x0
        v_pred = self.compute_velocity(model, xt, t, cond_emb)
        return F.mse_loss(v_pred, v_true, reduction="none").mean([1, 2, 3])
