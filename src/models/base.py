"""Base classes and interfaces for SDXL model implementations with extreme speedups."""
from enum import Enum, auto
from random import Random
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from abc import ABC, abstractmethod
import torch
import torch.backends.cudnn
from torch import Tensor


# Force maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

class ModelType(str, Enum):
    """Valid model types."""
    @classmethod
    def _missing_(cls, value: str) -> Optional["ModelType"]:
        """Handle case-insensitive lookup."""
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None

    BASE = "base"
    INPAINTING = "inpainting"
    REFINER = "refiner"
    SDXL = "sdxl"


class TimestepBiasStrategy(Enum):
    """Strategies for biasing timestep sampling during training."""
    NONE = "none"
    EARLIER = "earlier"
    LATER = "later" 
    RANGE = "range"


class BaseModelEmbedding:
    def __init__(
        self,
        uuid: str,
        token_count: int,
        placeholder: str,
    ):
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
    def __init__(self, model_type: ModelType):
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Invalid model type: {model_type}")
        self.model_type = model_type
        self.training = True
        self._device = torch.device("cuda")
        
        # Initialize embedding processor - will be set when encoders are initialized
        self.embedding_processor = None

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def generate_timestep_weights(
        self,
        num_timesteps: int,
        bias_strategy: TimestepBiasStrategy = TimestepBiasStrategy.NONE,
        bias_portion: float = 0.25,
        bias_multiplier: float = 2.0,
        bias_begin: Optional[int] = None,
        bias_end: Optional[int] = None
    ) -> torch.Tensor:
        """Generate weighted timestep sampling distribution.
        
        Args:
            num_timesteps: Total number of timesteps
            bias_strategy: Strategy for biasing timesteps
            bias_portion: Portion of timesteps to bias when using earlier/later strategies
            bias_multiplier: Weight multiplier for biased timesteps
            bias_begin: Starting timestep for range strategy
            bias_end: Ending timestep for range strategy
            
        Returns:
            Tensor of timestep weights normalized to sum to 1
        """
        pass

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
