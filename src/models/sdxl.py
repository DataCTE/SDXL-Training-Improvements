"""StableDiffusionXL model implementation with support for embeddings and LoRA."""
from contextlib import nullcontext
from random import Random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

import logging
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline as BasePipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    device_equals,
    torch_gc,
    torch_sync
)
from src.models.encoders.vae import VAEEncoder
from src.core.types import DataType
from src.models.base import BaseModel, BaseModelEmbedding, ModelType
from src.models.encoders.clip import encode_clip
from src.models.adapters.lora import LoRAModuleWrapper, AdditionalEmbeddingWrapper

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
            placeholder=placeholder if placeholder else f"<embedding-{uuid}>"
        )

        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector


class StableDiffusionXLPipeline(BasePipeline):
    """Custom SDXL pipeline implementation."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
    ):
        """Initialize custom SDXL pipeline.
        
        Args:
            vae: VAE model
            text_encoder: First text encoder
            text_encoder_2: Second text encoder
            tokenizer: First tokenizer
            tokenizer_2: Second tokenizer
            unet: UNet model
            scheduler: Noise scheduler
        """
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )

class StableDiffusionXLModel(torch.nn.Module, BaseModel):
    """StableDiffusionXL model with training optimizations."""

    def __init__(self, model_type: ModelType):
        """Initialize SDXL model.
        
        Args:
            model_type: Type of model (base, inpainting, etc)
        """
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_type=model_type)

        # Training state
        self.training = True

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
        """Move VAE to device with optimized CUDA transfer."""
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            # Ensure CUDA device is properly initialized
            torch.cuda.set_device(device)
        
        if not tensors_match_device(self.vae.state_dict(), device):
            with create_stream_context(torch.cuda.current_stream()):
                # Use CUDA streams for efficient transfer
                tensors_to_device_(self.vae.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.vae.state_dict())

    def text_encoder_to(self, device: torch.device) -> None:
        """Move both text encoders to device with explicit CUDA support."""
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            # Initialize CUDA device
            torch.cuda.set_device(device)
        
        with create_stream_context(torch.cuda.current_stream()):
            # Move encoders with CUDA optimization
            for encoder in [self.text_encoder_1, self.text_encoder_2]:
                if not device_equals(encoder.device, device):
                    tensors_to_device_(encoder.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), encoder.state_dict())

            # Move LoRA modules if present
            for lora in [self.text_encoder_1_lora, self.text_encoder_2_lora]:
                if lora is not None:
                    tensors_to_device_(lora.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), lora.state_dict())

    def text_encoder_1_to(self, device: torch.device) -> None:
        """Move first text encoder to device with optimized transfer."""
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_1.device, device):
                tensors_to_device_(self.text_encoder_1.state_dict(), device, non_blocking=True)
            if self.text_encoder_1_lora is not None:
                tensors_to_device_(self.text_encoder_1_lora.state_dict(), device, non_blocking=True)
            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_1.state_dict())

    def text_encoder_2_to(self, device: torch.device) -> None:
        """Move second text encoder to device with optimized transfer."""
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_2.device, device):
                tensors_to_device_(self.text_encoder_2.state_dict(), device, non_blocking=True)
            if self.text_encoder_2_lora is not None:
                tensors_to_device_(self.text_encoder_2_lora.state_dict(), device, non_blocking=True)
            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_2.state_dict())

    def unet_to(self, device: torch.device) -> None:
        """Move UNet to device with explicit CUDA/GPU support."""
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            # Initialize CUDA device
            torch.cuda.set_device(device)
        
        with create_stream_context(torch.cuda.current_stream()):
            # Move UNet with CUDA optimization
            if not device_equals(self.unet.device, device):
                tensors_to_device_(self.unet.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet.state_dict())
                    
            # Move LoRA if present
            if self.unet_lora is not None:
                tensors_to_device_(self.unet_lora.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet_lora.state_dict())

    def to(self, device: torch.device) -> None:
        """Move all model components to device with optimized transfer."""
        logger.info(f"Moving model components to {device}")
        
        # Use CUDA streams for pipelined transfers
        if device.type == "cuda":
            with create_stream_context(torch.cuda.current_stream()):
                self.vae_to(device)
                torch_gc()  # Clean up after VAE transfer
                
                self.text_encoder_to(device) 
                torch_gc()  # Clean up after encoder transfer
                
                self.unet_to(device)
                torch_gc()  # Final cleanup
                
            torch_sync()  # Ensure all transfers are complete
        else:
            # Sequential transfer for CPU
            self.vae_to(device)
            self.text_encoder_to(device)
            self.unet_to(device)

    def eval(self) -> None:
        """Set all model components to evaluation mode."""
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def train(self) -> None:
        """Set model components to training mode.
        Only UNet should be in training mode since we're only training it.
        """
        self.vae.eval()  # Keep VAE in eval mode
        self.text_encoder_1.eval()  # Keep text encoders in eval mode
        self.text_encoder_2.eval()
        self.unet.train()  # Only UNet in training mode

    def zero_grad(self) -> None:
        """Zero out gradients of trainable parameters."""
        self.unet.zero_grad(set_to_none=True)

    def parameters(self):
        """Get trainable parameters of the model.
        
        Returns:
            Iterator over model parameters
        """
        return self.unet.parameters()

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
        """Encode text using both SDXL text encoders."""
        # Tokenize if needed
        if tokens_1 is None and text is not None:
            tokens_1 = self.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer_1.model_max_length,
                return_tensors="pt"
            ).input_ids.to(self.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokens_2 = self.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt"
            ).input_ids.to(self.text_encoder_2.device)

        # Use optimized CLIP encoding from encoders.clip
        text_encoder_1_output, _ = encode_clip(
            text_encoder=self.text_encoder_1,
            tokens=tokens_1,
            default_layer=-2,
            layer_skip=text_encoder_1_layer_skip,
            text_encoder_output=text_encoder_1_output,
            add_pooled_output=False,
            use_attention_mask=False,
            add_layer_norm=False
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
            add_layer_norm=False
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
        text_encoder_output = torch.cat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output
