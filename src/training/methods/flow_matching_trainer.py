"""Flow Matching trainer implementation with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
import torch.nn.functional as F

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from typing import Dict, Optional, Tuple, Union
from src.core.types import DataType
from src.core.memory import torch_sync, create_stream_context
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class FlowMatchingTrainer(TrainingMethod):
    name = "flow_matching"

    if hasattr(torch, "compile"):
        def _compiled_loss(self, model, batch, generator=None):
            return self._compute_loss_impl(model, batch, generator)
        compute_loss = torch.compile(_compiled_loss, mode="reduce-overhead", fullgraph=True)
    else:
        compute_loss = None

    def compute_loss(self, model, batch, generator=None) -> Dict[str, torch.Tensor]:
        if self.compute_loss:
            return self.compute_loss(model, batch, generator)
        else:
            return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(self, model, batch, generator=None) -> Dict[str, torch.Tensor]:
        x1 = batch["model_input"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        t = self.sample_logit_normal((x1.shape[0],), x1.device, x1.dtype, generator=generator)
        x0 = torch.randn_like(x1)
        add_time_ids = get_add_time_ids(
            batch["original_sizes"],
            batch["crop_top_lefts"],
            batch["target_sizes"],
            dtype=prompt_embeds.dtype,
            device=x1.device
        )
        cond_emb = {
            "prompt_embeds": prompt_embeds,
            "added_cond_kwargs": {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
        }
        loss = self.compute_flow_matching_loss(self.unet, x0, x1, t, cond_emb)
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"]
        loss = loss.mean()
        torch_sync()
        return {"loss": loss}

    def sample_logit_normal(self, shape, device, dtype, mean=0.0, std=1.0, generator=None):
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        return torch.sigmoid(normal)

    def optimal_transport_path(self, x0, x1, t):
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_velocity(self, model, xt, t, cond_emb):
        v_pred = model(
            xt,
            t,
            encoder_hidden_states=cond_emb["prompt_embeds"],
            added_cond_kwargs=cond_emb["added_cond_kwargs"]
        ).sample
        return v_pred

    def compute_flow_matching_loss(self, model, x0, x1, t, cond_emb):
        xt = self.optimal_transport_path(x0, x1, t)
        v_true = x1 - x0
        v_pred = self.compute_velocity(model, xt, t, cond_emb)
        loss = F.mse_loss(v_pred, v_true, reduction="none").mean([1, 2, 3])
        return loss
