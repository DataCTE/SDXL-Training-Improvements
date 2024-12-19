"""Tensor and device management utilities."""
import gc
from collections.abc import Callable
from src.core.logging.logging import setup_logging
from contextlib import nullcontext
from packaging.version import Version
import packaging
import torch
import accelerate

logger = setup_logging(__name__, level="INFO")

accelerator = accelerate.Accelerator()
default_device = accelerator.device
torch_version = packaging.version.parse(torch.__version__)

def state_dict_has_prefix(state_dict: dict | None, prefix: str) -> bool:
    """Check if state dict has keys with prefix."""
    if not state_dict:
        return False
    return any(k.startswith(prefix) for k in state_dict)

def get_tensors(
    data: torch.Tensor | list | tuple | dict,
    include_parameter_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Get all tensors from nested data structure."""
    tensors = []

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        return [data.data]
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if i in include_parameter_indices or include_parameter_indices is None:
                tensors.extend(get_tensors(elem))
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensors.extend(get_tensors(elem))

    return tensors

def tensors_to_device_(
    data: torch.Tensor | list | tuple | dict,
    device: torch.device,
    include_parameter_indices: list[int] | None = None,
    non_blocking: bool = False,
    allocator: Callable[[torch.tensor], torch.tensor] | None = None,
) -> bool:
    """Move tensors to device in-place."""
    tensor_transferred = False

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if allocator is None:
            data.data = data.data.to(device=device, non_blocking=non_blocking)
        else:
            tensor = allocator(data)
            tensor.copy_(data, non_blocking=non_blocking)
            data.data = tensor
        tensor_transferred = True
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if i in include_parameter_indices or include_parameter_indices is None:
                tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)

    return tensor_transferred

def optimizer_to_device_(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state to device in-place."""
    for state in optimizer.state_dict()['state'].values():
        tensors_to_device_(state, device)

def replace_tensors_(
    target_data: torch.Tensor | list | tuple | dict,
    source_data: torch.Tensor | list | tuple | dict,
    include_parameter_indices: list[int] | None = None,
) -> None:
    """Replace tensor data in-place."""
    if isinstance(target_data, torch.Tensor) and include_parameter_indices is None:
        target_data.data = source_data.data
    elif isinstance(target_data, (list, tuple)):
        for i, elem in enumerate(target_data):
            if i in include_parameter_indices or include_parameter_indices is None:
                replace_tensors_(elem, source_data[i])
    elif isinstance(target_data, dict) and include_parameter_indices is None:
        for key, elem in target_data.items():
            replace_tensors_(elem, source_data[key])

def tensors_match_device(
    data: torch.Tensor | list | tuple | dict,
    device: torch.device,
    include_parameter_indices: list[int] | None = None,
) -> bool:
    """Check if all tensors match target device."""
    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if not device_equals(data.device, device):
            return False
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                if not tensors_match_device(elem, device):
                    return False
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            if not tensors_match_device(elem, device):
                return False

    return True

def tensors_record_stream(
    stream: torch.Stream,
    data: torch.Tensor | list | tuple | dict,
    include_parameter_indices: list[int] | None = None,
) -> None:
    """Record stream for CUDA tensors."""
    if isinstance(data, torch.Tensor):
        if data.device.type == "cuda":
            data.record_stream(stream)
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensors_record_stream(stream, elem, [])
    elif isinstance(data, dict):
        for elem in data.values():
            tensors_record_stream(stream, elem)

def unpin_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unpin module tensors."""
    for param in module.parameters():
        if param.is_pinned():
            param.data = param.clone()
    for buffer in module.buffers():
        if buffer.is_pinned():
            buffer.data = buffer.clone()
    return module

def device_equals(device1: torch.device, device2: torch.device) -> bool:
    """Check if two devices are equivalent."""
    return device1 is not None and device2 is not None \
        and device1.type == device2.type \
        and (0 if device1.index is None else device1.index) == (0 if device2.index is None else device2.index)

def torch_gc() -> None:
    """Garbage collect PyTorch memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Host memory cleanup handled by Python's GC

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def torch_sync() -> None:
    """Synchronize PyTorch devices."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

def create_stream_context(stream: torch.cuda.Stream) -> torch.cuda.StreamContext | nullcontext:
    """Create appropriate stream context."""
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.StreamContext(stream)
    return nullcontext()

def pin_tensor_(x: torch.Tensor) -> None:
    """Pin tensor memory (CUDA only)."""
    if torch.cuda.is_available():
        if not x.is_pinned():
            try:
                cudart = torch.cuda.cudart()
                err = cudart.cudaHostRegister(
                    x.data_ptr(),
                    x.numel() * x.element_size(),
                    cudart.cudaHostRegisterDefault
                )
                if err.value != 0:
                    raise RuntimeError(f"CUDA Error while trying to pin memory. error: {err.value}, ptr: {x.data_ptr()}, size: {x.numel() * x.element_size()}")
            except Exception as e:
                logger.warning(f"Failed to pin tensor memory: {str(e)}")

def unpin_tensor_(x: torch.Tensor) -> None:
    """Unpin tensor memory (CUDA only)."""
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostUnregister(x.data_ptr())
        if err.value != 0:
            raise RuntimeError(f"CUDA Error while trying to unpin memory. error {err.value}, ptr: {x.data_ptr()}")
