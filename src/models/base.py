"""Base model classes and types for SDXL."""
from enum import Enum, auto
from typing import Optional
import torch
from torch import Tensor

class ModelType(Enum):
    """Types of SDXL models."""
    BASE = auto()
    INPAINTING = auto()
    REFINER = auto()

class BaseModelEmbedding:
    """Base class for model embeddings."""
    
    def __init__(
        self,
        uuid: str,
        token_count: int,
        placeholder: str,
    ):
        """Initialize base embedding.
        
        Args:
            uuid: Unique identifier
            token_count: Number of tokens
            placeholder: Placeholder token
        """
        self.uuid = uuid
        self.token_count = token_count
        self.placeholder = placeholder

class BaseModel:
    """Base class for SDXL models."""
    
    def __init__(self, model_type: ModelType):
        """Initialize base model.
        
        Args:
            model_type: Type of model
        """
        self.model_type = model_type
        
    def to(self, device: torch.device) -> None:
        """Move model to device."""
        raise NotImplementedError
        
    def eval(self) -> None:
        """Set model to evaluation mode."""
        raise NotImplementedError
