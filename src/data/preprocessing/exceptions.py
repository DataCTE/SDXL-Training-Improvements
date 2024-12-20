"""Custom exceptions for preprocessing pipeline."""

class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass

class DataLoadError(PreprocessingError):
    """Raised when data loading fails."""
    pass

class PipelineConfigError(PreprocessingError):
    """Raised when pipeline configuration is invalid."""
    pass

class GPUProcessingError(PreprocessingError):
    """Raised when GPU processing fails."""
    pass

class CacheError(PreprocessingError):
    """Raised when cache operations fail."""
    pass

class DtypeError(PreprocessingError):
    """Raised when dtype conversion or validation fails."""
    pass

class DALIError(PreprocessingError):
    """Raised when DALI pipeline operations fail."""
    pass
