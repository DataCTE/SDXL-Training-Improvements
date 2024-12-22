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

def state_dict_has_prefix(state_dict: Optional[Dict[str, Any]], prefix: str) -> bool:
    """Check if state dict has keys with prefix.
    
    Args:
        state_dict: Model state dictionary
        prefix: Prefix to check for
        
    Returns:
        bool indicating if prefix exists
        
    Raises:
        TensorError: If state dict validation fails
    """
    with tensor_operation_context("state_dict_check"):
        try:
            if not state_dict:
                return False
                
            # Validate inputs
            if not isinstance(prefix, str):
                raise ValueError(f"Prefix must be a string, got {type(prefix)}")
                
            # Check for prefix in keys
            return any(k.startswith(prefix) for k in state_dict)
            
        except Exception as e:
            error_context = {
                'prefix': prefix,
                'state_dict_type': type(state_dict),
                'has_keys': bool(state_dict and state_dict.keys())
            }
            if not isinstance(e, ValueError):
                raise TensorError("Failed to check state dict prefix", error_context) from e
            raise


def optimizer_to_device_(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state to device with optimized memory handling.
    
    Args:
        optimizer: PyTorch optimizer
        device: Target device
        
    Raises:
        TensorError: If optimizer state transfer fails
    """
    with tensor_operation_context("optimizer_transfer"):
        try:
            state_dict = optimizer.state_dict()
            if 'state' not in state_dict:
                return
                
            # Use stream for transfer if CUDA
            stream = torch.cuda.Stream() if device.type == 'cuda' else None
            
            try:
                with create_stream_context(stream):
                    tensors_to_device_(state_dict['state'], device, non_blocking=True)
                    
                    # Record stream for tensor states
                    if stream:
                        for state in state_dict['state'].values():
                            if isinstance(state, dict):
                                for value in state.values():
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
    """Replace tensor data in-place with memory optimization.
    
    Args:
        target_data: Target data structure
        source_data: Source data structure 
        include_parameter_indices: Optional indices to include
        
    Raises:
        TensorError: If tensor replacement fails
    """
    with tensor_operation_context("replace_tensors"):
        try:
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
                        
        except Exception as e:
            error_context = {
                'target_type': type(target_data),
                'source_type': type(source_data),
                'indices': include_parameter_indices
            }
            raise TensorError("Failed to replace tensor data", error_context) from e

def tensors_match_device(
    data: Union[torch.Tensor, List, Tuple, Dict],
    device: torch.device,
    include_parameter_indices: Optional[List[int]] = None
) -> bool:
    """Check if tensors match target device with validation.
    
    Args:
        data: Data structure to check
        device: Target device
        include_parameter_indices: Optional indices to include
        
    Returns:
        bool indicating if tensors match device
    """
    with tensor_operation_context("device_check"):
        try:
            if isinstance(data, torch.Tensor) and include_parameter_indices is None:
                return device_equals(data.device, device)
            elif isinstance(data, (list, tuple)):
                return all(
                    tensors_match_device(elem, device)
                    for i, elem in enumerate(data)
                    if include_parameter_indices is None or i in include_parameter_indices
                )
            elif isinstance(data, dict) and include_parameter_indices is None:
                return all(
                    tensors_match_device(elem, device)
                    for elem in data.values()
                )
            return True
            
        except Exception as e:
            error_context = {
                'data_type': type(data),
                'device': str(device),
                'indices': include_parameter_indices
            }
            raise TensorError("Failed to check tensor devices", error_context) from e

def unpin_module(module: torch.nn.Module) -> torch.nn.Module:
    """Unpin module tensors with cleanup.
    
    Args:
        module: PyTorch module
        
    Returns:
        Unpinned module
        
    Raises:
        TensorError: If unpinning fails
    """
    with tensor_operation_context("unpin_module"):
        try:
            # Unpin parameters
            for param in module.parameters():
                if param.is_pinned():
                    param.data = param.data.clone()
                    unpin_tensor_(param)
                    
            # Unpin buffers
            for buffer in module.buffers():
                if buffer.is_pinned():
                    buffer.data = buffer.data.clone()
                    unpin_tensor_(buffer)
                    
            return module
            
        except Exception as e:
            error_context = {
                'module_type': type(module).__name__,
                'num_parameters': sum(1 for _ in module.parameters()),
                'num_buffers': sum(1 for _ in module.buffers())
            }
            raise TensorError("Failed to unpin module", error_context) from e

def torch_sync() -> None:
    """Synchronize all PyTorch devices.
    
    Raises:
        TensorError: If synchronization fails
    """
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
    if stream is None:
        yield nullcontext()
        return

    if not isinstance(stream, torch.cuda.Stream):
        raise StreamError(
            "Invalid stream type",
            context={'type': type(stream).__name__}
        )

    try:
        with torch.cuda.stream(stream):
            yield
    except Exception as e:
        error_context = {
            'stream_id': id(stream) if stream else None,
            'device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        }
        raise StreamError(f"Stream context creation failed: {error_context}") from e
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
