"""Memory optimization utilities for training with extreme speedups and efficient memory management."""
import torch
from typing import Dict, Optional, TYPE_CHECKING
from pathlib import Path


if TYPE_CHECKING:
    from src.data.config import Config

from ..types import DataType, ModelWeightDtypes
from src.core.logging import get_logger, LogConfig

logger = get_logger(__name__)

def setup_memory_optimizations(
    model: Optional[torch.nn.Module] = None,
    config: Optional["Config"] = None,  # type: ignore
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None
) -> bool:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')

    if config is None or device is None:
        return False
    try:
        if device.type != "cuda":
            logger.info("Memory optimizations only available for CUDA devices")
            return False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if model is not None and config.training.gradient_checkpointing:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            else:
                logger.warning("Model does not support gradient checkpointing")
        if config.training.mixed_precision:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
        if config.training.memory.enable_24gb_optimizations:
            logger.info("Setting up 24GB VRAM optimizations")
            if model is not None:
                if config.training.memory.layer_offload_fraction > 0:
                    model.layer_offload_fraction = config.training.memory.layer_offload_fraction
                    model.temp_device = torch.device(config.training.memory.temp_device)
                if config.training.memory.enable_activation_offloading:
                    model.enable_activation_offloading = True
                    model.enable_async_offloading = config.training.memory.enable_async_offloading
            if batch_size and micro_batch_size and batch_size > 1:
                logger.info(
                    f"Reducing effective batch size from {batch_size} to {micro_batch_size} "
                    f"with gradient accumulation"
                )
        return True
    except Exception as e:
        logger.error(f"Failed to setup memory optimizations: {str(e)}")
        return False

def verify_memory_optimizations(
    model: torch.nn.Module,
    config: "Config",  # type: ignore
    device: torch.device,
    logger: Optional["UnifiedLogger"] = None
) -> Dict[str, bool]:
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
