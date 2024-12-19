"""Memory optimization utilities for training."""
import logging
import torch
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def setup_memory_optimizations(
    model: torch.nn.Module,
    config: "Config",
    device: torch.device,
    batch_size: int,
    micro_batch_size: int
) -> bool:
    """Setup memory optimizations for training.
    
    Args:
        model: Model to optimize
        config: Training config
        device: Target device
        batch_size: Training batch size
        micro_batch_size: Micro batch size for gradient accumulation
        
    Returns:
        bool: Whether optimizations were successful
    """
    try:
        if device.type != "cuda":
            logger.info("Memory optimizations only available for CUDA devices")
            return False
            
        # Configure gradients
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable gradient checkpointing if configured
        if config.training.gradient_checkpointing:
            model.enable_gradient_checkpointing()
            
        # Configure automatic mixed precision
        if config.training.mixed_precision:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup memory optimizations: {str(e)}")
        return False

def verify_memory_optimizations(
    model: torch.nn.Module,
    config: "Config",
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Dict[str, bool]:
    """Verify memory optimizations are active.
    
    Args:
        model: Model to verify
        config: Training config
        device: Target device
        logger: Optional logger
        
    Returns:
        Dict of optimization states
    """
    states = {
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type == "cuda",
        "channels_last": (
            model.training and 
            next(model.parameters()).is_contiguous(memory_format=torch.channels_last)
        ),
        "gradient_checkpointing": (
            hasattr(model, "is_gradient_checkpointing") and
            model.is_gradient_checkpointing
        ),
        "mixed_precision": (
            config.training.mixed_precision and
            torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        )
    }
    
    if logger:
        for name, state in states.items():
            logger.info(f"{name}: {'enabled' if state else 'disabled'}")
            
    return states
