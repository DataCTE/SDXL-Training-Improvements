"""Memory optimization utilities for training."""
import logging
import torch
from src.core.logging.logging import setup_logging
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
from ..types import DataType, ModelWeightDtypes

logger = setup_logging(__name__, level="INFO")
from src.data.utils.paths import convert_windows_path, is_wsl

if TYPE_CHECKING:
    from ...data.config import Config

def setup_memory_optimizations(
    model: Optional[torch.nn.Module] = None,
    config: Optional["Config"] = None,  # type: ignore
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None
) -> bool:
    """Setup memory optimizations for training.
    
    Args:
        model: Model to optimize (optional)
        config: Training config (optional)
        device: Target device (optional)
        batch_size: Training batch size
        micro_batch_size: Micro batch size for gradient accumulation
        
    Returns:
        bool: Whether optimizations were successful
    """
    # Early return if no config or device
    if config is None or device is None:
        return False

    """Setup memory optimizations for training."""
    try:
        if device.type != "cuda":
            logger.info("Memory optimizations only available for CUDA devices")
            return False

        # Configure gradients
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable gradient checkpointing if configured and model is provided
        if model is not None and config.training.gradient_checkpointing:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            else:
                logger.warning("Model does not support gradient checkpointing")

        # Configure automatic mixed precision and throughput optimizations
        if config.training.mixed_precision:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

        # Setup 24GB VRAM optimizations if enabled
        if config.training.memory.enable_24gb_optimizations:
            logger.info("Setting up 24GB VRAM optimizations")
            
            # Configure layer offloading if model is provided
            if model is not None:
                if config.training.memory.layer_offload_fraction > 0:
                    model.layer_offload_fraction = config.training.memory.layer_offload_fraction
                    model.temp_device = torch.device(config.training.memory.temp_device)
                    
                # Enable activation offloading if configured
                if config.training.memory.enable_activation_offloading:
                    model.enable_activation_offloading = True
                    model.enable_async_offloading = config.training.memory.enable_async_offloading

            # Adjust batch size and gradient accumulation for memory constraints
            if batch_size and micro_batch_size and batch_size > 1:
                logger.info(f"Reducing effective batch size from {batch_size} to {micro_batch_size} with gradient accumulation")
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup memory optimizations: {str(e)}")
        return False

def verify_memory_optimizations(
    model: torch.nn.Module,
    config: "Config",  # type: ignore
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
            model is not None and
            model.training and 
            any(p.is_contiguous(memory_format=torch.channels_last) for p in model.parameters())
        ),
        "cuda_amp_autocast": torch.is_autocast_enabled(),
        "gradient_checkpointing": (
            model is not None and
            hasattr(model, "is_gradient_checkpointing") and
            model.is_gradient_checkpointing
        ),
        "mixed_precision": (
            config.training.mixed_precision and
            torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        ),
        "vram_optimizations": (
            config.training.memory.enable_24gb_optimizations and
            config.training.memory.layer_offload_fraction > 0
        ),
        "activation_offloading": (
            config.training.memory.enable_24gb_optimizations and
            config.training.memory.enable_activation_offloading
        ),
        "async_offloading": (
            config.training.memory.enable_24gb_optimizations and
            config.training.memory.enable_async_offloading
        )
    }
    
    if logger:
        for name, state in states.items():
            logger.info(f"{name}: {'enabled' if state else 'disabled'}")
            
    return states
