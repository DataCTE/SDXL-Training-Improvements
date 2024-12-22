"""Base classes and interfaces for SDXL model implementations."""
from enum import Enum, auto
from random import Random
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from abc import ABC, abstractmethod
import torch
from torch import Tensor

class ModelType(Enum):
    """Types of SDXL models."""
    BASE = auto()
    INPAINTING = auto()
    REFINER = auto()
    SDXL = auto()

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
        if not uuid:
            raise ValueError("UUID must not be empty")
        if token_count <= 0:
            raise ValueError("Token count must be positive")

        self.uuid = uuid
        self.token_count = token_count
        self.placeholder = placeholder if placeholder else f"<embedding-{uuid}>"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(uuid='{self.uuid}', token_count={self.token_count})"

class BaseModel(ABC):
    """Abstract base class defining the interface for SDXL models."""
    
    def __init__(self, model_type: ModelType):
        """Initialize base model.
        
        Args:
            model_type: Type of model
        """
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
            
        self.model_type = model_type
        self.training = True
        self._device = torch.device("cuda")

    @property
    def device(self) -> torch.device:
        """Get current device of model.
        
        Returns:
            Current torch device
        """
        return self._device

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Move all model components to device with optimized transfer.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        self._device = device

    @abstractmethod
    def vae_to(self, device: torch.device) -> None:
        """Move VAE to device.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        pass

    @abstractmethod
    def text_encoder_to(self, device: torch.device) -> None:
        """Move text encoders to device with CUDA optimization.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        pass

    @abstractmethod
    def text_encoder_1_to(self, device: torch.device) -> None:
        """Move first text encoder to device with optimized transfer.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        pass

    @abstractmethod
    def text_encoder_2_to(self, device: torch.device) -> None:
        """Move second text encoder to device with optimized transfer.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        pass

    @abstractmethod
    def unet_to(self, device: torch.device) -> None:
        """Move UNet to device.
        
        Args:
            device: Target device
            
        Raises:
            RuntimeError: If CUDA is not available when required
        """
        pass

    @abstractmethod
    def eval(self) -> None:
        """Set model components to evaluation mode."""
        self.training = False

    @abstractmethod
    def train(self) -> None:
        """Set model components to training mode."""
        self.training = True

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out gradients of trainable parameters."""
        pass

    @abstractmethod
    def parameters(self) -> Iterator[Tensor]:
        """Get trainable parameters.
        
        Returns:
            Iterator over model parameters
        """
        pass

    @abstractmethod
    def encode_text(
        self,
        train_device: torch.device,
        batch_size: int,
        rand: Optional[Random] = None,
        text: Optional[str] = None,
        tokens_1: Optional[Tensor] = None, 
        tokens_2: Optional[Tensor] = None,
        text_encoder_1_layer_skip: int = 0,
        text_encoder_2_layer_skip: int = 0,
        text_encoder_1_output: Optional[Tensor] = None,
        text_encoder_2_output: Optional[Tensor] = None,
        text_encoder_1_dropout_probability: Optional[float] = None,
        text_encoder_2_dropout_probability: Optional[float] = None,
        pooled_text_encoder_2_output: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Encode text using both SDXL text encoders with optimized processing.
        
        Args:
            train_device: Device to run encoding on
            batch_size: Size of batch being processed
            rand: Optional random number generator for dropout
            text: Optional text to encode
            tokens_1: Optional pre-tokenized input for encoder 1 
            tokens_2: Optional pre-tokenized input for encoder 2
            text_encoder_1_layer_skip: Number of layers to skip in encoder 1
            text_encoder_2_layer_skip: Number of layers to skip in encoder 2
            text_encoder_1_output: Optional cached output from encoder 1
            text_encoder_2_output: Optional cached output from encoder 2
            text_encoder_1_dropout_probability: Optional dropout prob for encoder 1
            text_encoder_2_dropout_probability: Optional dropout prob for encoder 2
            pooled_text_encoder_2_output: Optional cached pooled output
            
        Returns:
            Tuple of (combined_encoder_output, pooled_encoder_2_output)
            
        Raises:
            ValueError: If both text and tokens are None
        """
        pass