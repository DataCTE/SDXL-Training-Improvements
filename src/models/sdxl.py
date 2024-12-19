"""StableDiffusionXL model implementation with support for embeddings and LoRA."""
import logging
from contextlib import nullcontext
from random import Random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from .base import BaseModel, BaseModelEmbedding, ModelType
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .base import BaseModel, BaseModelEmbedding, ModelType
from .encoders.clip import encode_clip
from .adapters.lora import LoRAModuleWrapper, AdditionalEmbeddingWrapper
from ..core.types import DataType

logger = logging.getLogger(__name__)

class StableDiffusionXLModelEmbedding(BaseModelEmbedding):
    """Embedding model for SDXL with dual encoder support."""
    
    def __init__(
        self,
        uuid: str,
        text_encoder_1_vector: Tensor,
        text_encoder_2_vector: Tensor,
        placeholder: str,
    ):
        """Initialize SDXL embedding.
        
        Args:
            uuid: Unique identifier
            text_encoder_1_vector: First text encoder embedding
            text_encoder_2_vector: Second text encoder embedding
            placeholder: Placeholder token
        """
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder,
        )

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector


class StableDiffusionXLModel(BaseModel):
    """StableDiffusionXL model with training optimizations."""

    def __init__(self, model_type: ModelType):
        """Initialize SDXL model.
        
        Args:
            model_type: Type of model (base, inpainting, etc)
        """
        super().__init__(model_type=model_type)

        # Base model components
        self.tokenizer_1: Optional[CLIPTokenizer] = None
        self.tokenizer_2: Optional[CLIPTokenizer] = None
        self.noise_scheduler: Optional[DDIMScheduler] = None
        self.text_encoder_1: Optional[CLIPTextModel] = None
        self.text_encoder_2: Optional[CLIPTextModelWithProjection] = None
        self.vae: Optional[AutoencoderKL] = None
        self.unet: Optional[UNet2DConditionModel] = None

        # Autocast contexts
        self.autocast_context = nullcontext()
        self.vae_autocast_context = nullcontext()

        # Data types
        self.train_dtype = DataType.FLOAT_32
        self.vae_train_dtype = DataType.FLOAT_32

        # Embedding training data
        self.embedding: Optional[StableDiffusionXLModelEmbedding] = None
        self.embedding_state: Optional[Tuple[Tensor, Tensor]] = None
        self.additional_embeddings: List[StableDiffusionXLModelEmbedding] = []
        self.additional_embedding_states: List[Optional[Tuple[Tensor, Tensor]]] = []
        self.embedding_wrapper_1: Optional[AdditionalEmbeddingWrapper] = None
        self.embedding_wrapper_2: Optional[AdditionalEmbeddingWrapper] = None

        # LoRA training data
        self.text_encoder_1_lora: Optional[LoRAModuleWrapper] = None
        self.text_encoder_2_lora: Optional[LoRAModuleWrapper] = None
        self.unet_lora: Optional[LoRAModuleWrapper] = None
        self.lora_state_dict: Optional[Dict] = None

        # Configuration
        self.sd_config: Optional[Dict] = None
        self.sd_config_filename: Optional[str] = None

    def vae_to(self, device: torch.device) -> None:
        """Move VAE to device."""
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device) -> None:
        """Move both text encoders to device."""
        self.text_encoder_1.to(device=device)
        self.text_encoder_2.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def text_encoder_1_to(self, device: torch.device) -> None:
        """Move first text encoder to device."""
        self.text_encoder_1.to(device=device)

        if self.text_encoder_1_lora is not None:
            self.text_encoder_1_lora.to(device)

    def text_encoder_2_to(self, device: torch.device) -> None:
        """Move second text encoder to device."""
        self.text_encoder_2.to(device=device)

        if self.text_encoder_2_lora is not None:
            self.text_encoder_2_lora.to(device)

    def unet_to(self, device: torch.device) -> None:
        """Move UNet to device."""
        self.unet.to(device=device)

        if self.unet_lora is not None:
            self.unet_lora.to(device)

    def to(self, device: torch.device) -> None:
        """Move all model components to device."""
        self.vae_to(device)
        self.text_encoder_to(device)
        self.unet_to(device)

    def eval(self) -> None:
        """Set all model components to evaluation mode."""
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        """Create SDXL pipeline from model components."""
        return StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        """Add trained embeddings to prompt text."""
        return self._add_embeddings_to_prompt(
            self.additional_embeddings,
            self.embedding,
            prompt
        )

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
        """Encode text using both SDXL text encoders.
        
        Args:
            train_device: Target device for training
            batch_size: Batch size
            rand: Random number generator
            text: Input text to encode
            tokens_1: Pre-tokenized input for encoder 1
            tokens_2: Pre-tokenized input for encoder 2
            text_encoder_1_layer_skip: Layers to skip in encoder 1
            text_encoder_2_layer_skip: Layers to skip in encoder 2
            text_encoder_1_output: Cached encoder 1 output
            text_encoder_2_output: Cached encoder 2 output
            text_encoder_1_dropout_probability: Dropout prob for encoder 1
            text_encoder_2_dropout_probability: Dropout prob for encoder 2
            pooled_text_encoder_2_output: Cached pooled output
            
        Returns:
            Tuple of (combined encoder output, pooled output)
        """
        # Tokenize if needed
        if tokens_1 is None and text is not None:
            tokenizer_output = self.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokenizer_output = self.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(self.text_encoder_2.device)

        # Encode with both encoders
        text_encoder_1_output, _ = encode_clip(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-2,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            add_pooled_output=False,
            use_attention_mask=False,
            add_layer_norm=False,
        )

        text_encoder_2_output, pooled_text_encoder_2_output = encode_clip(
            text_encoder=self.text_encoder_2,
            tokens=tokens_2,
            default_layer=-2,
            layer_skip=text_encoder_2_layer_skip,
            text_encoder_output=text_encoder_2_output,
            add_pooled_output=True,
            pooled_text_encoder_output=pooled_text_encoder_2_output,
            use_attention_mask=False,
            add_layer_norm=False,
        )

        # Apply dropout if configured
        if text_encoder_1_dropout_probability is not None:
            dropout_text_encoder_1_mask = torch.tensor(
                [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                device=train_device
            ).float()
            text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask[:, None, None]

        if text_encoder_2_dropout_probability is not None:
            dropout_text_encoder_2_mask = torch.tensor(
                [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                device=train_device
            ).float()
            pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask[:, None]
            text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask[:, None, None]

        # Combine encoder outputs
        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output
