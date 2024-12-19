"""Memory optimization utilities for training."""
import logging
import torch
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.config import Config

logger = logging.getLogger(__name__)

def configure_model_memory_format(
    model: torch.nn.Module,
) -> None:
    """Configure model memory format for training.
    
    Args:
        model: Model to configure
        config: Training config
    """
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
