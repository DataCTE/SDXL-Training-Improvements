"""Memory optimization utilities for training with extreme speedups."""
import logging
import torch
from typing import Optional, TYPE_CHECKING

# Force maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

if TYPE_CHECKING:
    from ...data.config import Config

logger = logging.getLogger(__name__)

def configure_model_memory_format(model: torch.nn.Module) -> None:
    """Configure model memory format for training with channels-last optimization."""
    if torch.cuda.is_available():
        model.to(memory_format=torch.channels_last)

# Optionally compile for further speedups
if hasattr(torch, "compile"):
    configure_model_memory_format = torch.compile(
        configure_model_memory_format, mode="reduce-overhead", fullgraph=False
    )
