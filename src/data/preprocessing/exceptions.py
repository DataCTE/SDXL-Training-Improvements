"""Custom exceptions for preprocessing pipeline with detailed error context."""
from typing import Optional, Any, Dict
from src.core.logging.logging import setup_logging

logger = setup_logging(__name__)

class PreprocessingError(Exception):
    """Base exception for preprocessing errors with enhanced context tracking."""
    
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None):
        """Initialize with message and optional context dictionary."""
        super().__init__(message)
        self.context = context or {}
        self.message = message
        # Log error with context
        logger.error(self.format_error())
        
    def format_error(self) -> str:
        """Format error message with detailed context."""
        base_msg = self.message
        if self.context:
            context_str = "\nContext:\n" + "\n".join(
                f"  {k}: {v}" for k, v in self.context.items()
            )
            return f"{base_msg}{context_str}"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'context': self.context
        }

    def __str__(self) -> str:
        """Format error message with context if available."""
        return self.format_error()

class DataLoadError(PreprocessingError):
    """Raised when data loading fails.
    
    Contexts:
        - file_path: Path to file that failed to load
        - batch_size: Size of batch being processed
        - data_type: Type of data being loaded
    """
    pass

class PipelineConfigError(PreprocessingError):
    """Raised when pipeline configuration is invalid.
    
    Contexts:
        - config_key: Configuration key that failed validation
        - invalid_value: The invalid value
        - expected_type: Expected type/format
    """
    pass

class GPUProcessingError(PreprocessingError):
    """Raised when GPU processing fails.
    
    Contexts:
        - device_id: ID of GPU device
        - cuda_version: CUDA version
        - memory_allocated: Current GPU memory usage
        - operation: Operation that failed
    """
    pass

class CacheError(PreprocessingError):
    """Raised when cache operations fail.
    
    Contexts:
        - cache_dir: Cache directory path
        - operation: Cache operation that failed
        - file_size: Size of file being cached
    """
    pass

class DtypeError(PreprocessingError):
    """Raised when dtype conversion or validation fails.
    
    Contexts:
        - source_dtype: Original dtype
        - target_dtype: Desired dtype
        - tensor_shape: Shape of tensor
    """
    pass

class DALIError(PreprocessingError):
    """Raised when DALI pipeline operations fail.
    
    Contexts:
        - pipeline_id: DALI pipeline identifier
        - operation: DALI operation that failed
        - batch_size: Size of batch being processed
    """
    pass

class TensorValidationError(PreprocessingError):
    """Raised when tensor validation fails.
    
    Contexts:
        - expected_shape: Expected tensor shape
        - actual_shape: Actual tensor shape
        - expected_dtype: Expected dtype
        - actual_dtype: Actual dtype
    """
    pass

class StreamError(PreprocessingError):
    """Raised when CUDA stream operations fail.
    
    Contexts:
        - stream_id: CUDA stream identifier
        - operation: Stream operation that failed
        - device_id: GPU device ID
    """
    pass

class MemoryError(PreprocessingError):
    """Raised when memory operations fail.
    
    Contexts:
        - operation: Memory operation that failed
        - allocated: Current memory allocation
        - requested: Requested memory
        - available: Available memory
    """
    pass
