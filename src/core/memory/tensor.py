"""Enhanced tensor and device management utilities with improved memory handling and error recovery."""
import gc
from collections.abc import Callable
from contextlib import nullcontext, contextmanager
from typing import Union, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import weakref
from packaging.version import Version
import packaging
import torch
import accelerate
from src.core.logging.logging import setup_logging

logger = setup_logging(__name__, level="INFO")

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

class StreamError(TensorError):
    """Raised for CUDA stream errors."""
    pass

@contextmanager
def tensor_operation_context(operation_name: str):
    """Context manager for tensor operations with error handling."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        yield
    except Exception as e:
        error_context = {
            'operation': operation_name,
            'cuda_available': torch.cuda.is_available(),
            'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        logger.error(f"Tensor operation failed: {str(e)}", extra=error_context)
        raise TensorError(f"Failed during {operation_name}", error_context) from e
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            tensor_stats.memory_allocated = end_memory
            tensor_stats.peak_memory = max(tensor_stats.peak_memory, end_memory)

def get_tensors(
    data: Union[torch.Tensor, List, Tuple, Dict],
    include_parameter_indices: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """Enhanced tensor extraction with validation."""
    with tensor_operation_context("get_tensors"):
        tensors = []
        try:
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
            return tensors
        except Exception as e:
            raise TensorError("Failed to extract tensors", {'data_type': type(data)}) from e

def tensors_to_device_(
    data: Union[torch.Tensor, List, Tuple, Dict],
    device: torch.device,
    include_parameter_indices: Optional[List[int]] = None,
    non_blocking: bool = False,
    allocator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> bool:
    """Enhanced tensor device transfer with memory optimization."""
    with tensor_operation_context("tensors_to_device"):
        try:
            tensor_transferred = False
            
            if isinstance(data, torch.Tensor):
                if include_parameter_indices is None:
                    # Create CUDA stream for transfer if applicable
                    stream = torch.cuda.Stream() if torch.cuda.is_available() and device.type == 'cuda' else None
                    
                    try:
                        with torch.cuda.stream(stream) if stream else nullcontext():
                            if allocator is None:
                                # Optimize memory format for device
                                if device.type == 'cuda':
                                    data.data = data.data.to(
                                        device=device,
                                        memory_format=torch.channels_last,
                                        non_blocking=non_blocking
                                    )
                                else:
                                    data.data = data.data.to(device=device, non_blocking=non_blocking)
                            else:
                                tensor = allocator(data)
                                tensor.copy_(data, non_blocking=non_blocking)
                                data.data = tensor
                                
                            if stream:
                                tensors_record_stream(stream, data)
                                
                    finally:
                        if stream:
                            stream.synchronize()
                            
                    tensor_transferred = True
                    tensor_stats.device_transfers += 1
                    
            elif isinstance(data, (list, tuple)):
                for i, elem in enumerate(data):
                    if include_parameter_indices is None or i in include_parameter_indices:
                        tensor_transferred |= tensors_to_device_(
                            elem, device, non_blocking=non_blocking, allocator=allocator
                        )
            elif isinstance(data, dict):
                if include_parameter_indices is None:
                    for elem in data.values():
                        tensor_transferred |= tensors_to_device_(
                            elem, device, non_blocking=non_blocking, allocator=allocator
                        )
                        
            return tensor_transferred
            
        except Exception as e:
            error_context = {
                'device': str(device),
                'data_type': type(data),
                'non_blocking': non_blocking
            }
            raise DeviceError("Failed to transfer tensors to device", error_context) from e

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
                'data_type': type(data)
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
    """Enhanced tensor unpinning with validation."""
    with tensor_operation_context("unpin_tensor"):
        try:
            if torch.cuda.is_available() and x.is_pinned():
                cudart = torch.cuda.cudart()
                err = cudart.cudaHostUnregister(x.data_ptr())
                if err.value != 0:
                    raise MemoryError(f"CUDA error {err.value}")
                tensor_stats.unpin_operations += 1
        except Exception as e:
            error_context = {
                'tensor_shape': tuple(x.shape),
                'tensor_dtype': str(x.dtype),
                'device': str(x.device)
            }
            raise MemoryError("Failed to unpin tensor memory", error_context) from e

def device_equals(device1: torch.device, device2: torch.device) -> bool:
    """Enhanced device comparison with validation."""
    try:
        if device1 is None or device2 is None:
            return False
        return (device1.type == device2.type and 
                (0 if device1.index is None else device1.index) == 
                (0 if device2.index is None else device2.index))
    except Exception as e:
        error_context = {
            'device1': str(device1),
            'device2': str(device2)
        }
        raise DeviceError("Failed to compare devices", error_context) from e

@contextmanager
def create_stream_context(stream: Optional[torch.cuda.Stream] = None) -> Union[torch.cuda.StreamContext, nullcontext]:
    """Enhanced stream context with automatic cleanup."""
    if not isinstance(stream, torch.cuda.Stream):
        yield nullcontext()
        return
        
    try:
        with torch.cuda.stream(stream):
            yield torch.cuda.StreamContext(stream)
    except Exception as e:
        error_context = {
            'stream_id': id(stream) if stream else None,
            'device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        }
        raise StreamError("Stream context creation failed", error_context) from e
    finally:
        if stream is not None:
            stream.synchronize()

def torch_gc() -> None:
    """Enhanced garbage collection with memory tracking."""
    with tensor_operation_context("garbage_collection"):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            prev_mem = (torch.cuda.memory_allocated() 
                       if torch.cuda.is_available() else 0)
                
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
                
            curr_mem = (torch.cuda.memory_allocated() 
                       if torch.cuda.is_available() else 0)
                       
            if curr_mem < prev_mem:
                logger.debug(f"Memory freed: {(prev_mem - curr_mem) / 1024**2:.2f}MB")
                
        except Exception as e:
            error_context = {
                'cuda_available': torch.cuda.is_available(),
                'mps_available': torch.backends.mps.is_available()
            }
            raise MemoryError("Garbage collection failed", error_context) from e

def get_tensor_stats() -> TensorStats:
    """Get current tensor operation statistics."""
    return tensor_stats

def reset_tensor_stats() -> None:
    """Reset tensor operation statistics."""
    global tensor_stats
    tensor_stats = TensorStats()