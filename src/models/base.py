"""Base classes and interfaces for SDXL model implementations with extreme speedups."""
from enum import Enum, auto
from random import Random
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from abc import ABC, abstractmethod
import torch
import torch.backends.cudnn
from torch import Tensor
from .encoders.embedding import TextEmbeddingProcessor

# Force maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

class ModelType(Enum):
    BASE = auto()
    INPAINTING = auto()
    REFINER = auto()
    SDXL = auto()

from .embeddings import BaseModelEmbedding

class BaseModel(ABC):
    def __init__(self, model_type: ModelType):
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
        self.model_type = model_type
        self.training = True
        self._device = torch.device("cuda")
        
        # Initialize embedding processor
        self.embedding_processor = TextEmbeddingProcessor(
            device=self._device,
            dtype=self._dtype.to_torch_dtype(),
            enable_memory_tracking=True
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def to(self, device: torch.device) -> None:
        self._device = device

    @abstractmethod
    def vae_to(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def text_encoder_to(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def text_encoder_1_to(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def text_encoder_2_to(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def unet_to(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        self.training = False

    @abstractmethod
    def train(self) -> None:
        self.training = True

    @abstractmethod
    def zero_grad(self) -> None:
        pass

    @abstractmethod
    def parameters(self) -> Iterator[Tensor]:
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
        pass
