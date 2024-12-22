"""Shared utilities for preprocessing."""
import logging
import torch
from typing import Dict, Optional, Union, Any
from contextlib import nullcontext

from src.core.memory.tensor import (
    tensors_to_device_,
    create_stream_context,
    tensors_record_stream,
    torch_sync,
    pin_tensor_,
    unpin_tensor_,
    device_equals
)

logger = logging.getLogger(__name__)

def process_tensor_batch(
    tensor: torch.Tensor,
    device: Union[str, torch.device],
    use_pinned_memory: bool = True,
    enable_memory_tracking: bool = True,
    memory_stats: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """Process tensor batch with optimized memory handling.
    
    Args:
        tensor: Input tensor
        device: Target device
        use_pinned_memory: Whether to use pinned memory
        enable_memory_tracking: Whether to track memory usage
        memory_stats: Optional dict to track memory stats
        
    Returns:
        Processed tensor
    """
    try:
        # Track memory usage
        if enable_memory_tracking and memory_stats is not None:
            current = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_stats['current_allocated'] = current
            memory_stats['peak_allocated'] = max(
                memory_stats.get('peak_allocated', 0),
                current
            )

        # Pin memory if configured
        if use_pinned_memory:
            pin_tensor_(tensor)

        try:
            # Get CUDA streams if available
            compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

            # Process with streams
            with create_stream_context(transfer_stream) if transfer_stream else nullcontext():
                # Move to device if needed
                if not device_equals(tensor.device, device):
                    tensors_to_device_(tensor, device, non_blocking=True)
                if transfer_stream:
                    tensors_record_stream(transfer_stream, tensor)

            with create_stream_context(compute_stream) if compute_stream else nullcontext():
                if compute_stream:
                    compute_stream.wait_stream(transfer_stream)
                    tensors_record_stream(compute_stream, tensor)

            return tensor

        finally:
            # Cleanup
            if use_pinned_memory:
                unpin_tensor_(tensor)
            torch_sync()

    except Exception as e:
        logger.error(f"Failed to process tensor batch: {str(e)}")
        raise

def validate_tensor(
    tensor: torch.Tensor,
    expected_dims: int = 4,
    enable_memory_tracking: bool = True,
    memory_stats: Optional[Dict[str, Any]] = None
) -> bool:
    """Validate tensor properties with memory tracking.
    
    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        enable_memory_tracking: Whether to track memory usage
        memory_stats: Optional dict to track memory stats
        
    Returns:
        bool indicating if tensor is valid
    """
    try:
        if enable_memory_tracking and memory_stats is not None:
            current = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_stats['current_allocated'] = current
            memory_stats['peak_allocated'] = max(
                memory_stats.get('peak_allocated', 0),
                current
            )

        if not isinstance(tensor, torch.Tensor):
            return False

        if tensor.dim() != expected_dims:
            return False

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False

        # Check tensor device placement
        if tensor.device.type == 'cuda':
            # Ensure tensor is in contiguous memory
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

        return True

    except Exception as e:
        logger.error(f"Tensor validation failed: {str(e)}")
        return False
