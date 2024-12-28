"""Enhanced tensor and device management utilities with improved memory handling and error recovery."""
import gc
from collections.abc import Callable
from contextlib import nullcontext, contextmanager
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import packaging
import torch
import accelerate
from src.core.logging.logging import setup_logging

logger = setup_logging(
    module_name=__name__,
    level="INFO",
    log_dir="outputs/logs"  # Add default log directory
)

# Initialize accelerator and device
accelerator = accelerate.Accelerator()
default_device = accelerator.device
torch_version = packaging.version.parse(torch.__version__)

@dataclass
class TensorStats:
    """Track tensor operation statistics."""
    pin_operations: int = 0
    unpin_operations: int = 0
    device_transfers: int = 0
    stream_records: int = 0
    memory_allocated: int = 0
    peak_memory: int = 0

# Global statistics tracker
tensor_stats = TensorStats()

class TensorError(Exception):
    """Base exception for tensor operations."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

class DeviceError(TensorError):
    """Raised for device-related errors."""
    pass

class MemoryError(TensorError):
    """Raised for memory-related errors."""
    pass

class StreamError(Exception):
    """Exception for stream-related errors."""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}

@contextmanager
def create_stream_context(stream: Optional[torch.cuda.Stream] = None) -> Union[torch.cuda.StreamContext, nullcontext]:
    """Create a CUDA stream context with proper device management and synchronization."""
    if stream is None or not torch.cuda.is_available():
        with nullcontext() as nc:
            yield nc
        return

    try:
        # Create a new stream if needed
        if stream is None:
            stream = torch.cuda.Stream()

        # Ensure previous operations are complete
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(stream) as stream_ctx:
            try:
                yield stream_ctx
            finally:
                if stream is not None:
                    stream.synchronize()

    except Exception as e:
        error_context = {
            'stream_id': id(stream) if stream else None,
            'device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
            'cuda_available': torch.cuda.is_available(),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        raise StreamError("Stream context creation failed", error_context) from e

@contextmanager
def tensor_operation_context(operation_name: str):
    """Context manager for tensor operations with enhanced memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        yield
    except Exception as e:
        error_context = {
            'operation': operation_name,
            'cuda_available': torch.cuda.is_available(),
            'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
        logger.error(f"Tensor operation failed: {str(e)}", extra=error_context)
        raise TensorError(f"Failed during {operation_name}", error_context) from e
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def get_tensors(
    data: Union[torch.Tensor, List, Tuple, Dict],
    include_parameter_indices: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """Enhanced tensor extraction with validation."""
    with tensor_operation_context("get_tensors"):
        tensors = []
        if isinstance(data, torch.Tensor):
            if include_parameter_indices is None:
                return [data.data]
        elif isinstance(data, (list, tuple)):
            for i, elem in enumerate(data):
                if include_parameter_indices is None or i in include_parameter_indices:
                    tensors.extend(get_tensors(elem))
        elif isinstance(data, dict):
            if include_parameter_indices is None:
                for elem in data.values():
                    tensors.extend(get_tensors(elem))
        elif data is not None:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return tensors

def tensors_to_device_(data: Union[torch.Tensor, Dict, List], device: torch.device, non_blocking: bool = True) -> None:
    """Move tensors to device with enhanced error handling and memory tracking."""
    with tensor_operation_context("tensor_transfer"):
        try:
            if isinstance(data, torch.Tensor):
                if data.is_pinned() and device.type == "cuda":
                    data = data.to(device, non_blocking=non_blocking)
                    if non_blocking:
                        with create_stream_context(torch.cuda.current_stream()) as stream:
                            data.record_stream(stream)
                else:
                    data = data.to(device, non_blocking=False)
                tensor_stats.device_transfers += 1
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, (torch.Tensor, dict, list)):
                        tensors_to_device_(value, device, non_blocking)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (torch.Tensor, dict, list)):
                        tensors_to_device_(item, device, non_blocking)
        except Exception as e:
            error_context = {
                'data_type': type(data),
                'device': str(device),
                'non_blocking': non_blocking,
                'cuda_available': torch.cuda.is_available(),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
            raise StreamError("Failed to transfer tensors to device", error_context) from e

def state_dict_has_prefix(state_dict: Optional[Dict[str, Any]], prefix: str) -> bool:
    """Check if state dict has keys with prefix."""
    with tensor_operation_context("state_dict_check"):
        if not state_dict:
            return False
        if not isinstance(prefix, str):
            raise ValueError(f"Prefix must be a string, got {type(prefix)}")
        return any(k.startswith(prefix) for k in state_dict)

def optimizer_to_device_(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state to device with optimized memory handling."""
    with tensor_operation_context("optimizer_transfer"):
        try:
            state_dict = optimizer.state_dict()
            if 'state' not in state_dict:
                return
            stream = torch.cuda.Stream() if device.type == 'cuda' else None
            try:
                with create_stream_context(stream):
                    tensors_to_device_(state_dict['state'], device, non_blocking=True)
                    if stream:
                        for st in state_dict['state'].values():
                            if isinstance(st, dict):
                                for value in st.values():
                                    if isinstance(value, torch.Tensor):
                                        tensors_record_stream(stream, value)
            finally:
                if stream:
                    stream.synchronize()
        except Exception as e:
            error_context = {
                'optimizer_type': type(optimizer).__name__,
                'device': str(device),
                'has_state': 'state' in getattr(optimizer, 'state_dict', lambda: {})()
            }
            raise TensorError("Failed to move optimizer to device", error_context) from e

def replace_tensors_(
    target_data: Union[torch.Tensor, List, Tuple, Dict],
    source_data: Union[torch.Tensor, List, Tuple, Dict],
    include_parameter_indices: Optional[List[int]] = None
) -> None:
    """Replace tensor data in-place with memory optimization."""
    with tensor_operation_context("replace_tensors"):
        if isinstance(target_data, torch.Tensor) and include_parameter_indices is None:
            if not isinstance(source_data, torch.Tensor):
                raise TypeError(f"Source must be tensor, got {type(source_data)}")
            target_data.data = source_data.data
        elif isinstance(target_data, (list, tuple)):
            if not isinstance(source_data, (list, tuple)):
                raise TypeError(f"Source must be sequence, got {type(source_data)}")
            for i, (target_elem, source_elem) in enumerate(zip(target_data, source_data)):
                if include_parameter_indices is None or i in include_parameter_indices:
                    replace_tensors_(target_elem, source_elem)
        elif isinstance(target_data, dict) and include_parameter_indices is None:
            if not isinstance(source_data, dict):
                raise TypeError(f"Source must be dict, got {type(source_data)}")
            for key, target_elem in target_data.items():
                if key in source_data:
                    replace_tensors_(target_elem, source_data[key])

def tensors_match_device(
    data: Union[torch.Tensor, List, Tuple, Dict],
    device: torch.device,
    include_parameter_indices: Optional[List[int]] = None
) -> bool:
    """Check if tensors match target device."""
    with tensor_operation_context("device_check"):
        if isinstance(data, torch.Tensor) and include_parameter_indices is None:
            return device_equals(data.device, device)
        elif isinstance(data, (list, tuple)):
            return all(
                tensors_match_device(elem, device)
                for i, elem in enumerate(data)
                if include_parameter_indices is None or i in include_parameter_indices
            )
        elif isinstance(data, dict) and include_parameter_indices is None:
            return all(tensors_match_device(elem, device) for elem in data.values())
        return True

def unpin_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unpin module tensors with cleanup."""
    with tensor_operation_context("unpin_module"):
        for param in module.parameters():
            if param.is_pinned():
                param.data = param.data.clone()
                unpin_tensor_(param)
        for buffer in module.buffers():
            if buffer.is_pinned():
                buffer.data = buffer.data.clone()
                unpin_tensor_(buffer)
        return module

def torch_gc() -> None:
    """Enhanced garbage collection with memory tracking."""
    with tensor_operation_context("garbage_collection"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            error_context = {
                'cuda_available': torch.cuda.is_available(),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
            raise MemoryError("Garbage collection failed", error_context) from e

def torch_sync() -> None:
    """Synchronize all PyTorch devices."""
    with tensor_operation_context("sync"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        except Exception as e:
            error_context = {
                'cuda_available': torch.cuda.is_available(),
                'mps_available': torch.backends.mps.is_available()
            }
            raise TensorError("Failed to synchronize devices", error_context) from e

def tensors_record_stream(
    stream: torch.cuda.Stream,
    data: Union[torch.Tensor, List, Tuple, Dict],
    include_parameter_indices: Optional[List[int]] = None
) -> None:
    """Enhanced stream recording with validation."""
    with tensor_operation_context("record_stream"):
        try:
            if isinstance(data, torch.Tensor):
                if data.device.type == "cuda":
                    data.record_stream(stream)
                    tensor_stats.stream_records += 1
            elif isinstance(data, (list, tuple)):
                for i, elem in enumerate(data):
                    if include_parameter_indices is None or i in include_parameter_indices:
                        tensors_record_stream(stream, elem)
            elif isinstance(data, dict):
                for elem in data.values():
                    tensors_record_stream(stream, elem)
        except Exception as e:
            error_context = {
                'stream_id': id(stream),
                'data_type': type(data),
                'cuda_available': torch.cuda.is_available()
            }
            raise StreamError("Failed to record stream", error_context) from e

def pin_tensor_(x: torch.Tensor) -> None:
    """Enhanced tensor pinning with error handling."""
    with tensor_operation_context("pin_tensor"):
        try:
            if torch.cuda.is_available() and not x.is_pinned():
                x.pin_memory()
                tensor_stats.pin_operations += 1
        except Exception as e:
            error_context = {
                'tensor_shape': tuple(x.shape),
                'tensor_dtype': str(x.dtype),
                'device': str(x.device)
            }
            raise MemoryError("Failed to pin tensor memory", error_context) from e

def unpin_tensor_(x: torch.Tensor) -> None:
    """Enhanced tensor unpinning with validation and memory management."""
    with tensor_operation_context("unpin_tensor"):
        try:
            if torch.cuda.is_available() and x.is_pinned():
                torch.cuda.current_stream().synchronize()
                cudart = torch.cuda.cudart()
                err = cudart.cudaHostUnregister(x.data_ptr())
                if err.value != 0:
                    error_msg = f"cuda error {err.value}"
                    if hasattr(cudart, 'cudaGetErrorString'):
                        error_msg = cudart.cudaGetErrorString(err).decode()
                    raise MemoryError(error_msg)
                tensor_stats.unpin_operations += 1
        except Exception as e:
            error_context = {
                'tensor_shape': tuple(x.shape),
                'tensor_dtype': str(x.dtype),
                'device': str(x.device),
                'is_pinned': x.is_pinned() if hasattr(x, 'is_pinned') else None
            }
            raise MemoryError("Failed to unpin tensor memory", error_context) from e

def device_equals(device1: torch.device, device2: torch.device) -> bool:
    """Enhanced device comparison with validation."""
    try:
        if device1 is None or device2 is None:
            return False
        return (
            device1.type == device2.type and
            (0 if device1.index is None else device1.index) ==
            (0 if device2.index is None else device2.index)
        )
    except Exception as e:
        error_context = {'device1': str(device1), 'device2': str(device2)}
        raise DeviceError("Failed to compare devices", error_context) from e

def get_tensor_stats() -> TensorStats:
    """Get current tensor operation statistics."""
    return tensor_stats

def reset_tensor_stats() -> None:
    """Reset tensor operation statistics."""
    global tensor_stats
    tensor_stats = TensorStats()
