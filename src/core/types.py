"""Common type definitions."""
from enum import Enum, auto

class DataType(Enum):
    """Data types for model components."""
    FLOAT_32 = auto()
    FLOAT_16 = auto()
    BFLOAT_16 = auto()
