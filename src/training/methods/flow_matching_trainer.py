"""Flow Matching trainer implementation with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
from src.training.methods.base import make_picklable
from src.core.history import TorchHistory
import torch.nn.functional as F


from typing import Dict, Optional, Tuple, Union
from src.core.types import DataType
from src.core.memory import torch_sync, create_stream_context
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class FlowMatchingTrainer(TrainingMethod):
    name = "flow_matching"

    def __init__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        super().__init__(*args, **kwargs)
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
        if hasattr(torch, "compile"):
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('history', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
    @make_picklable 
    def compute_loss(self, model, batch, generator=None) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(model, batch, generator)
        return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        try:
            print("\n=== Starting Flow Matching Loss Computation ===")
            
            # Validate batch contents
            print("\nValidating Batch Contents:")
            required_keys = {
                'model_input', 'prompt_embeds', 'pooled_prompt_embeds',
                'original_sizes', 'crop_top_lefts', 'target_sizes'
            }
            available_keys = set(batch.keys())
            print(f"Available batch keys: {available_keys}")
            print(f"Required batch keys: {required_keys}")
            
            missing_keys = required_keys - available_keys
            if missing_keys:
                error_msg = f"Missing required keys in batch: {missing_keys}"
                print(f"\nERROR: {error_msg}")
                print("\nBatch contents summary:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    elif isinstance(value, list):
                        print(f"  {key}: list of length {len(value)}")
                    else:
                        print(f"  {key}: type={type(value)}")
                raise KeyError(error_msg)

            # Access validated input
            x1 = batch["model_input"]
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            
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
            
            # Apply loss weights if present
            if "loss_weights" in batch:
                loss = loss * batch["loss_weights"]
            loss = loss.mean()
            
            torch_sync()
            return {"loss": loss}

        except Exception as e:
            logger.error(f"Error computing Flow Matching loss: {str(e)}", exc_info=True)
            torch_sync()
            raise

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
