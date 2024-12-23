"""StableDiffusionXL model implementation with support for embeddings and LoRA."""
from contextlib import nullcontext
from random import Random
from typing import Dict, List, Optional, Tuple, Union

import gc
import torch
from torch import Tensor

import logging
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline as BasePipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    device_equals,
    torch_sync,
    torch_gc,
    create_stream_context,
    tensors_record_stream
)
from src.models.encoders.vae import VAEEncoder
from src.core.types import DataType, ModelWeightDtypes
from src.models.base import BaseModel, BaseModelEmbedding, ModelType
from src.models.encoders.clip import encode_clip
from src.models.encoders.vae import VAEEncoder
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
        scheduler: DDPMScheduler
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
            model_type: Type of SDXL model (BASE, INPAINTING, or REFINER)
        """
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_type)
        
        # Initialize model components
        self.vae = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = None
        
        # Initialize LoRA components
        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None
        
        # Model configuration
        self.model_type = model_type
        self._dtype = DataType.FLOAT_32  # Default dtype

    @property
    def dtype(self) -> DataType:
        """Get model dtype."""
        return self._dtype

    @dtype.setter 
    def dtype(self, value: Union[str, DataType]) -> None:
        """Set model dtype.
        
        Args:
            value: New dtype value as string or DataType
        """
        if isinstance(value, str):
            value = DataType.from_str(value)
        elif not isinstance(value, DataType):
            raise ValueError(f"Invalid dtype: {value}")
        self._dtype = value

    def from_pretrained(
        self,
        pretrained_model_name: str,
        dtype: Union[DataType, str] = DataType.FLOAT_32,
        use_safetensors: bool = True,
        **kwargs
    ) -> None:
        """Load pretrained model components.
        
        Args:
            pretrained_model_name: HuggingFace model name or path
            dtype: Base data type for model weights
            use_safetensors: Whether to use safetensors format
            **kwargs: Additional arguments for from_pretrained
        """
        try:
            logger.info(f"Loading model components from {pretrained_model_name}")
            
            # Set model dtype
            if isinstance(dtype, str):
                dtype = DataType.from_str(dtype)
            self.dtype = dtype
            
            # Create weight dtypes configuration
            model_dtypes = ModelWeightDtypes(
                train_dtype=dtype,
                fallback_train_dtype=DataType.FLOAT_32,
                unet=dtype,
                prior=dtype,
                text_encoder=dtype,
                text_encoder_2=dtype,
                vae=dtype,
                effnet_encoder=dtype,
                decoder=dtype,
                decoder_text_encoder=dtype,
                decoder_vqgan=dtype,
                lora=dtype,
                embedding=dtype
            )
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
                subfolder="vae",
                torch_dtype=model_dtypes.vae.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
            
            # Load text encoders
            self.text_encoder_1 = CLIPTextModel.from_pretrained(
                pretrained_model_name,
                subfolder="text_encoder",
                torch_dtype=model_dtypes.text_encoder.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                pretrained_model_name,
                subfolder="text_encoder_2",
                torch_dtype=model_dtypes.text_encoder_2.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
            
            # Load UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=model_dtypes.unet.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
            
            # Load tokenizers
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                pretrained_model_name,
                subfolder="tokenizer"
            )
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                pretrained_model_name,
                subfolder="tokenizer_2"
            )
            
            # Initialize scheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_name,
                subfolder="scheduler"
            )
            
            logger.info("Successfully loaded all model components")
            
        except Exception as e:
            error_msg = f"Failed to load pretrained model: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def vae_to(self, device: torch.device) -> None:
        """Move VAE to device with optimized CUDA transfer."""
        if isinstance(self.vae, VAEEncoder):
            self.vae.to(device=device)
        else:
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("cuda is not available")
                # Ensure CUDA device is properly initialized
                device_idx = device.index if device.index is not None else 0
                torch.cuda.set_device(device_idx)
            
            if not tensors_match_device(self.vae.state_dict(), device):
                with create_stream_context(torch.cuda.current_stream()):
                    # Use CUDA streams for efficient transfer
                    tensors_to_device_(self.vae.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), self.vae.state_dict())

    def text_encoder_to(self, device: torch.device) -> None:
        """Move both text encoders to device with explicit CUDA support."""
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("cuda is not available")
            # Initialize CUDA device with proper index handling
            device_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(device_idx)
        
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
                raise RuntimeError("cuda is not available")
            # Initialize CUDA device
            device_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(device_idx)
        
        with create_stream_context(torch.cuda.current_stream()):
            # Move UNet with CUDA optimization
            if not device_equals(self.unet.device, device):
                tensors_to_device_(self.unet.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet.state_dict())
                    
            # Move LoRA if present
            if self.unet_lora is not None:
                tensors_to_device_(self.unet_lora.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet_lora.state_dict())

    

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
    
    def to(self, device: torch.device) -> None:
        """Move all model components to device with optimized transfer."""
        logger.info(f"Moving model components to {device}")
        
        try:
            # Use CUDA streams for pipelined transfers
            if device.type == "cuda":
                # Create a new stream for transfers
                stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                
                with create_stream_context(stream):
                    self.vae_to(device)
                    torch_gc()  # Clean up after VAE transfer
                    
                    self.text_encoder_to(device) 
                    torch_gc()  # Clean up after encoder transfer
                    
                    self.unet_to(device)
                    torch_gc()  # Final cleanup
                    
                # Ensure all transfers are complete
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
                    
            else:
                # Sequential transfer for CPU
                self.vae_to(device)
                self.text_encoder_to(device)
                self.unet_to(device)
                
        except Exception as e:
            error_context = {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
            logger.error("Failed to move model to device", extra=error_context)
            raise
    
    def create_pipeline(self) -> 'StableDiffusionXLPipeline':
        """Create SDXL pipeline from current model components.
        
        Returns:
            StableDiffusionXLPipeline: Pipeline for inference
        """
        if not all([
            self.vae,
            self.text_encoder_1,
            self.text_encoder_2,
            self.tokenizer_1,
            self.tokenizer_2,
            self.unet,
            self.noise_scheduler
        ]):
            raise ValueError("Cannot create pipeline: some model components are not initialized")
            
        # Use local import to avoid circular dependencies
        
        
        return StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer_1=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler
        )
