"""Flow Matching implementation for SDXL training."""
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ..core.memory.tensor import create_stream_context, tensors_record_stream

logger = logging.getLogger(__name__)

def sample_logit_normal(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Sample from logit-normal distribution.
    
    Args:
        shape: Output tensor shape
        device: Target device
        dtype: Target dtype
        
    Returns:
        Sampled tensor in [0,1]
    """
    with create_stream_context(torch.cuda.current_stream()):
        normal = torch.randn(shape, device=device, dtype=dtype)
        tensors_record_stream(torch.cuda.current_stream(), normal)
    return torch.sigmoid(normal)

def optimal_transport_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """Compute optimal transport path between samples.
    
    Args:
        x0: Starting point
        x1: Target point
        t: Time values in [0,1]
        
    Returns:
        Points along optimal transport path
    """
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x0 + t * x1

def compute_flow_matching_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    condition_embeddings: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor:
    """Compute Flow Matching training loss.
    
    Args:
        model: UNet model
        x0: Starting point samples
        x1: Target samples
        t: Time values
        condition_embeddings: Optional conditioning embeddings
        
    Returns:
        Flow matching loss
    """
    # Get batch size
    batch_size = x0.shape[0]
    
    # Compute optimal transport path
    xt = optimal_transport_path(x0, x1, t)
    
    # True velocity field (dx/dt)
    v_true = x1 - x0
    
    # Predict velocity
    v_pred = model(
        xt,
        t,
        encoder_hidden_states=condition_embeddings["prompt_embeds"] if condition_embeddings else None,
        added_cond_kwargs=condition_embeddings["added_cond_kwargs"] if condition_embeddings else None
    ).sample
    
    # Compute MSE loss
    loss = F.mse_loss(v_pred, v_true, reduction="none")
    loss = loss.mean(dim=[1, 2, 3])  # Mean over dimensions except batch
    
    return loss.mean()  # Mean over batch
