"""StableDiffusionXL model implementation with extreme speedups and optimizations."""
import logging
import torch
import torch.backends.cudnn
from torch import Tensor
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline as BasePipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from .encoders.embedding import TextEmbeddingProcessor

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from contextlib import nullcontext
from random import Random
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import gc
from src.core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    device_equals,
    torch_sync,
    torch_gc,
    create_stream_context,
    tensors_record_stream,
    pin_tensor_,
    unpin_tensor_
)
from src.models.encoders.vae import VAEEncoder
from src.models.encoders.clip import CLIPEncoder
from src.core.types import DataType, ModelWeightDtypes
from src.models.base import BaseModel, BaseModelEmbedding, ModelType
from src.models.adapters.lora import LoRAModuleWrapper, AdditionalEmbeddingWrapper

from src.core.logging.logging import setup_logging

logger = setup_logging(__name__)

class ModelError(Exception):
    """Base exception for model errors."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message)
        self.context = context or {}
        logger.error(self.format_error())

    def format_error(self) -> str:
        """Format error message with context."""
        msg = str(self)
        if self.context:
            msg += "\nContext:\n" + "\n".join(f"  {k}: {v}" for k, v in self.context.items())
        return msg

class DeviceError(ModelError):
    """Raised for device-related errors."""
    pass

class EncodingError(ModelError):
    """Raised for encoding-related errors."""
    pass


class StableDiffusionXLModelEmbedding(BaseModelEmbedding):
    """Enhanced embedding implementation for SDXL."""
    def __init__(
        self,
        uuid: str,
        text_encoder_1_vector: Tensor,
        text_encoder_2_vector: Tensor,
        placeholder: str,
    ):
        if not isinstance(text_encoder_1_vector, Tensor) or not isinstance(text_encoder_2_vector, Tensor):
            raise ValueError("Embedding vectors must be tensors")
            
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder if placeholder else f"<embedding-{uuid}>"
        )
        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector

    def to(self, device: torch.device) -> 'StableDiffusionXLModelEmbedding':
        """Move embeddings to specified device."""
        self.text_encoder_1_vector = self.text_encoder_1_vector.to(device)
        self.text_encoder_2_vector = self.text_encoder_2_vector.to(device)
        return self


class StableDiffusionXLPipeline(BasePipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder_1: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_1: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )


class StableDiffusionXLModel(torch.nn.Module, BaseModel):
    """StableDiffusionXL model with training optimizations and memory handling."""

    def __init__(
        self, 
        model_type: ModelType,
        enable_memory_efficient_attention: bool = True,
        enable_vae_slicing: bool = False,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False
    ):
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_type)

        # Initialize components
        self.vae = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = None

        # Initialize LoRA adapters
        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None

        self.model_type = model_type
        self._dtype = DataType.FLOAT_32

        # Store optimization flags
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload

        # Initialize optimized encoder wrappers
        self.clip_encoder_1: Optional[CLIPEncoder] = None
        self.clip_encoder_2: Optional[CLIPEncoder] = None 
        self.vae_encoder: Optional[VAEEncoder] = None

        # Initialize memory tracking
        self.memory_stats = {
            'peak_allocated': 0,
            'current_allocated': 0,
            'num_allocations': 0
        }

        logger.info("Initialized SDXL model", extra={
            'model_type': str(model_type),
            'optimizations': {
                'memory_efficient_attention': enable_memory_efficient_attention,
                'vae_slicing': enable_vae_slicing,
                'model_cpu_offload': enable_model_cpu_offload,
                'sequential_cpu_offload': enable_sequential_cpu_offload
            }
        })

    @property
    def dtype(self) -> DataType:
        return self._dtype

    @dtype.setter
    def dtype(self, value: Union[str, DataType]) -> None:
        if isinstance(value, str):
            value = DataType.from_str(value)
        elif not isinstance(value, DataType):
            raise ValueError(f"Invalid dtype: {value}")
        self._dtype = value

    def from_pretrained(
        self,
        pretrained_model_name: str,
        dtype: Union[DataType, str, ModelWeightDtypes] = DataType.FLOAT_32,
        use_safetensors: bool = True,
        **kwargs
    ) -> None:
        """
        Load pretrained model components for SDXL.
        """
        logger.info(f"Loading model components from {pretrained_model_name}")

        try:
            # 1. Resolve dtype config
            if isinstance(dtype, str):
                base_dtype = DataType.from_str(dtype)
                model_dtypes = ModelWeightDtypes.from_single_dtype(base_dtype)
            elif isinstance(dtype, DataType):
                model_dtypes = ModelWeightDtypes.from_single_dtype(dtype)
            elif isinstance(dtype, ModelWeightDtypes):
                model_dtypes = dtype
            else:
                raise ValueError(f"Invalid dtype: {dtype}")

            self.dtype = model_dtypes.train_dtype

            # 2. Load VAE with error tracking
            logger.info("Loading VAE...")
            try:
                self.vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name,
                    subfolder="vae",
                    torch_dtype=model_dtypes.vae.to_torch_dtype(),
                    use_safetensors=use_safetensors
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load VAE: {str(e)}") from e

            # 3. Load text encoders with error tracking
            logger.info("Loading text encoder 1...")
            try:
                self.text_encoder_1 = CLIPTextModel.from_pretrained(
                    pretrained_model_name,
                    subfolder="text_encoder",
                    torch_dtype=model_dtypes.text_encoder.to_torch_dtype(),
                    use_safetensors=use_safetensors
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load text encoder 1: {str(e)}") from e

            logger.info("Loading text encoder 2...")
            try:
                self.text_encoder_2 = CLIPTextModel.from_pretrained(
                    pretrained_model_name,
                    subfolder="text_encoder_2",
                    torch_dtype=model_dtypes.text_encoder_2.to_torch_dtype(),
                    use_safetensors=use_safetensors
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load text encoder 2: {str(e)}") from e

            # 4. Load UNet with error tracking
            logger.info("Loading UNet...")
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name,
                    subfolder="unet",
                    torch_dtype=model_dtypes.unet.to_torch_dtype(),
                    use_safetensors=use_safetensors
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load UNet: {str(e)}") from e

            # 5. Load tokenizers
            logger.info("Loading tokenizer 1...")
            self.tokenizer_1 = CLIPTokenizer.from_pretrained(
                pretrained_model_name,
                subfolder="tokenizer"
            )
            logger.info("Loading tokenizer 2...")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                pretrained_model_name,
                subfolder="tokenizer_2"
            )

            # 6. Load scheduler
            logger.info("Loading noise scheduler...")
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_name,
                subfolder="scheduler"
            )

            # Initialize encoders using dedicated implementations
            self._initialize_encoders()

            logger.info("Successfully loaded all model components.")

        except Exception as e:
            error_context = {
                'error': str(e),
                'component': 'unknown'
            }
            if self.vae is None:
                error_context['component'] = 'vae'
            elif self.text_encoder_1 is None:
                error_context['component'] = 'text_encoder_1'
            elif self.text_encoder_2 is None:
                error_context['component'] = 'text_encoder_2'
            elif self.unet is None:
                error_context['component'] = 'unet'
            elif self.tokenizer_1 is None:
                error_context['component'] = 'tokenizer_1'
            elif self.tokenizer_2 is None:
                error_context['component'] = 'tokenizer_2'
            elif self.noise_scheduler is None:
                error_context['component'] = 'noise_scheduler'

            error_msg = f"Failed to load {error_context['component']}: {str(e)}"
            logger.error(error_msg, extra=error_context)
            raise ValueError(error_msg) from e

    def _initialize_encoders(self):
        """Initialize optimized encoder wrappers using imported implementations."""
        try:
            # Initialize VAE encoder using imported VAEEncoder
            self.vae_encoder = VAEEncoder(
                vae=self.vae,
                device=self.device,
                dtype=next(self.vae.parameters()).dtype
            )

            # Initialize CLIP encoders using imported CLIPEncoder
            self.clip_encoder_1 = CLIPEncoder(
                text_encoder=self.text_encoder_1,
                tokenizer=self.tokenizer_1,
                device=self.device,
                dtype=next(self.text_encoder_1.parameters()).dtype,
                enable_memory_efficient_attention=self.enable_memory_efficient_attention
            )
            
            self.clip_encoder_2 = CLIPEncoder(
                text_encoder=self.text_encoder_2,
                tokenizer=self.tokenizer_2,
                device=self.device,
                dtype=next(self.text_encoder_2.parameters()).dtype,
                enable_memory_efficient_attention=self.enable_memory_efficient_attention
            )

        except Exception as e:
            raise ModelError("Failed to initialize encoders", {
                'error': str(e),
                'device': str(self.device)
            })

        

    def vae_to(self, device: torch.device) -> None:
        """
        Move VAE to the specified device.
        """
        if isinstance(self.vae, VAEEncoder):
            self.vae.to(device=device)
        else:
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("cuda is not available")
                device_idx = device.index if device.index is not None else 0
                torch.cuda.set_device(device_idx)
            if not tensors_match_device(self.vae.state_dict(), device):
                with create_stream_context(torch.cuda.current_stream()):
                    tensors_to_device_(self.vae.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), self.vae.state_dict())

    def text_encoder_to(self, device: torch.device) -> None:
        """
        Move both text encoders and their wrappers to the specified device.
        """
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("cuda is not available")
            device_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(device_idx)

        with create_stream_context(torch.cuda.current_stream()):
            # Move base encoders
            for encoder in [self.text_encoder_1, self.text_encoder_2]:
                if not device_equals(encoder.device, device):
                    tensors_to_device_(encoder.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), encoder.state_dict())
            
            # Update CLIP encoder wrappers with new device
            if self.clip_encoder_1:
                self.clip_encoder_1.device = device
            if self.clip_encoder_2:
                self.clip_encoder_2.device = device

            # Move LoRA if present
            for lora in [self.text_encoder_1_lora, self.text_encoder_2_lora]:
                if lora is not None:
                    tensors_to_device_(lora.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), lora.state_dict())

    def text_encoder_1_to(self, device: torch.device) -> None:
        """
        Move only the first text encoder (and LoRA) to the specified device.
        """
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_1.device, device):
                tensors_to_device_(self.text_encoder_1.state_dict(), device, non_blocking=True)
            if self.text_encoder_1_lora is not None:
                tensors_to_device_(self.text_encoder_1_lora.state_dict(), device, non_blocking=True)

            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_1.state_dict())

    def text_encoder_2_to(self, device: torch.device) -> None:
        """
        Move only the second text encoder (and LoRA) to the specified device.
        """
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_2.device, device):
                tensors_to_device_(self.text_encoder_2.state_dict(), device, non_blocking=True)
            if self.text_encoder_2_lora is not None:
                tensors_to_device_(self.text_encoder_2_lora.state_dict(), device, non_blocking=True)

            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_2.state_dict())

    def unet_to(self, device: torch.device) -> None:
        """
        Move UNet (and LoRA) to the specified device.
        """
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("cuda is not available")
            device_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(device_idx)

        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.unet.device, device):
                tensors_to_device_(self.unet.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet.state_dict())

            if self.unet_lora is not None:
                tensors_to_device_(self.unet_lora.state_dict(), device, non_blocking=True)
                tensors_record_stream(torch.cuda.current_stream(), self.unet_lora.state_dict())

    def eval(self) -> None:
        """
        Set VAE and text encoders to eval mode, UNet to eval mode as well.
        """
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def train(self) -> None:
        """
        Keep VAE and text encoders in eval mode, set UNet to train mode.
        """
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.train()

    def zero_grad(self) -> None:
        """
        Zero out gradients of the UNet (trainable component).
        """
        self.unet.zero_grad(set_to_none=True)

    def parameters(self):
        """
        Return trainable parameters of the UNet by default.
        """
        return self.unet.parameters()

    def add_embeddings_to_prompt(self, prompt: str) -> str:
        """Inject custom embeddings into a prompt string.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Modified prompt with embedding tokens injected
            
        Example:
            If prompt is "a photo of a cat" and there's a custom embedding
            with placeholder "<style-modern>", the result might be:
            "<style-modern> a photo of a cat"
        """
        try:
            if not prompt:
                return prompt
                
            # Start with original prompt
            modified_prompt = prompt
            
            # Add LoRA embeddings if present
            if hasattr(self, 'additional_embeddings') and self.additional_embeddings:
                for embedding in self.additional_embeddings:
                    if isinstance(embedding, StableDiffusionXLModelEmbedding):
                        # Add placeholder token to prompt
                        modified_prompt = f"{embedding.placeholder} {modified_prompt}"
                        
            # Add base embedding if present
            if hasattr(self, 'embedding') and self.embedding:
                if isinstance(self.embedding, StableDiffusionXLModelEmbedding):
                    # Add base embedding placeholder to start
                    modified_prompt = f"{self.embedding.placeholder} {modified_prompt}"
                    
            # Remove extra whitespace
            modified_prompt = " ".join(modified_prompt.split())
            
            logger.debug("Modified prompt with embeddings", extra={
                'original_prompt': prompt,
                'modified_prompt': modified_prompt,
                'has_additional': hasattr(self, 'additional_embeddings'),
                'has_base': hasattr(self, 'embedding')
            })
            
            return modified_prompt
            
        except Exception as e:
            logger.error("Failed to add embeddings to prompt", extra={
                'error': str(e),
                'prompt': prompt,
                'stack_trace': True
            })
            # Return original prompt on error
            return prompt

    def _add_embeddings_to_prompt(
        self,
        additional_embeddings: Optional[List[StableDiffusionXLModelEmbedding]] = None,
        base_embedding: Optional[StableDiffusionXLModelEmbedding] = None,
        prompt: str = ""
    ) -> str:
        """Internal method to add embeddings to prompt with more control.
        
        Args:
            additional_embeddings: List of additional embeddings to inject
            base_embedding: Base embedding to inject
            prompt: Original prompt text
            
        Returns:
            Modified prompt with embedding tokens injected
        """
        try:
            if not prompt:
                return prompt
                
            # Start with original prompt
            modified_prompt = prompt
            
            # Add additional embeddings
            if additional_embeddings:
                for embedding in additional_embeddings:
                    if isinstance(embedding, StableDiffusionXLModelEmbedding):
                        modified_prompt = f"{embedding.placeholder} {modified_prompt}"
                        
            # Add base embedding
            if base_embedding and isinstance(base_embedding, StableDiffusionXLModelEmbedding):
                modified_prompt = f"{base_embedding.placeholder} {modified_prompt}"
                
            # Clean up whitespace
            modified_prompt = " ".join(modified_prompt.split())
            
            return modified_prompt
            
        except Exception as e:
            logger.error("Failed to add embeddings to prompt", extra={
                'error': str(e),
                'prompt': prompt,
                'stack_trace': True
            })
            return prompt

    def encode_clip(
        self,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
        tokens: torch.Tensor,
        default_layer: int = -2,
        layer_skip: int = 0,
        text_encoder_output: Optional[torch.Tensor] = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Optional[torch.Tensor] = None,
        use_attention_mask: bool = False,
        add_layer_norm: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode text using CLIP encoder with layer selection and pooling options.
        
        Args:
            text_encoder: CLIP text encoder model
            tokens: Input token ids
            default_layer: Default layer to use if no skip
            layer_skip: Number of layers to skip
            text_encoder_output: Optional pre-computed encoder output
            add_pooled_output: Whether to return pooled output
            pooled_text_encoder_output: Optional pre-computed pooled output
            use_attention_mask: Whether to use attention mask
            add_layer_norm: Whether to apply layer norm
            
        Returns:
            Tuple of (text embeddings, optional pooled embeddings)
        """
        if text_encoder_output is not None:
            return text_encoder_output, pooled_text_encoder_output
            
        outputs = text_encoder(
            tokens,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        if layer_skip > 0:
            text_encoder_output = outputs.hidden_states[-layer_skip]
        else:
            text_encoder_output = outputs.hidden_states[default_layer]
            
        # Get pooled output if requested
        if add_pooled_output:
            if pooled_text_encoder_output is not None:
                return text_encoder_output, pooled_text_encoder_output
            elif hasattr(outputs, "pooler_output"):
                pooled_text_encoder_output = outputs.pooler_output
            else:
                pooled_text_encoder_output = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled_text_encoder_output = None
            
        # Apply layer norm if requested
        if add_layer_norm:
            text_encoder_output = torch.nn.functional.layer_norm(
                text_encoder_output, 
                text_encoder_output.shape[-1:]
            )
            
        return text_encoder_output, pooled_text_encoder_output

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
        additional_embeddings: Optional[List[StableDiffusionXLModelEmbedding]] = None,
        base_embedding: Optional[StableDiffusionXLModelEmbedding] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode text with both text encoders (and optional dropout).
        """
        try:
            # Process text with embeddings if provided
            if text is not None:
                processed = self.embedding_processor.process_embeddings(
                    text,
                    additional_embeddings=additional_embeddings,
                    base_embedding=base_embedding
                )
                text = processed["processed_text"]

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

            # Encode with first text encoder
            text_encoder_1_output, _ = self.encode_clip(
                text_encoder=self.text_encoder_1,
                tokens=tokens_1,
                default_layer=-2,
                layer_skip=text_encoder_1_layer_skip,
                text_encoder_output=text_encoder_1_output,
                add_pooled_output=False,
                use_attention_mask=False,
                add_layer_norm=False
            )

            # Encode with second text encoder
            text_encoder_2_output, pooled_text_encoder_2_output = self.encode_clip(
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

            # Apply optional dropout to text_encoder_1 output
            if text_encoder_1_dropout_probability is not None:
                dropout_text_encoder_1_mask = torch.tensor(
                    [rand.random() > text_encoder_1_dropout_probability for _ in range(batch_size)],
                    device=train_device
                ).float()
                text_encoder_1_output = text_encoder_1_output * dropout_text_encoder_1_mask[:, None, None]

            # Apply optional dropout to text_encoder_2 output
            if text_encoder_2_dropout_probability is not None:
                dropout_text_encoder_2_mask = torch.tensor(
                    [rand.random() > text_encoder_2_dropout_probability for _ in range(batch_size)],
                    device=train_device
                ).float()
                pooled_text_encoder_2_output = pooled_text_encoder_2_output * dropout_text_encoder_2_mask[:, None]
                text_encoder_2_output = text_encoder_2_output * dropout_text_encoder_2_mask[:, None, None]

            # Concatenate final embeddings
            text_encoder_output = torch.cat([text_encoder_1_output, text_encoder_2_output], dim=-1)

            # Combine with custom embeddings if present
            if additional_embeddings or base_embedding:
                text_encoder_output = self.embedding_processor.combine_embeddings(
                    text_encoder_output,
                    additional_embeddings=additional_embeddings,
                    base_embedding=base_embedding
                )

            # Validate final embeddings
            if not self.embedding_processor.validate_embeddings({
                'text_encoder_output': text_encoder_output,
                'pooled_output': pooled_text_encoder_2_output
            }):
                raise ValueError("Invalid embeddings detected")

            return text_encoder_output, pooled_text_encoder_2_output

        except Exception as e:
            logger.error("Text encoding failed", extra={
                'error': str(e),
                'text': text,
                'has_additional': bool(additional_embeddings),
                'has_base': bool(base_embedding),
                'stack_trace': True
            })
            raise

    def to(self, device: torch.device) -> None:
        """
        Move all model components to the specified device (VAE, text encoders, UNet).
        """
        logger.info(f"Moving model components to {device}")
        try:
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available")

            # Create stream context only for CUDA devices
            stream_ctx = (
                create_stream_context(torch.cuda.current_stream())  # Use current stream instead of creating new one
                if device.type == "cuda"
                else nullcontext()
            )

            with stream_ctx:
                # Move components sequentially with memory cleanup and synchronization
                for component_name, move_fn in [
                    ("VAE", self.vae_to),
                    ("text encoders", self.text_encoder_to),
                    ("UNet", self.unet_to)
                ]:
                    try:
                        logger.debug(f"Moving {component_name} to {device}")
                        move_fn(device)
                        if device.type == "cuda":
                            torch_sync()  # Synchronize after each component
                            torch_gc()
                            logger.debug(f"Successfully moved {component_name} to {device}")
                    except Exception as comp_error:
                        raise RuntimeError(f"Failed to move {component_name} to {device}: {str(comp_error)}") from comp_error

                # Final synchronization
                if device.type == "cuda":
                    torch_sync()
                    logger.info(f"Successfully moved all components to {device}")

        except Exception as e:
            error_context = {
                'device': str(device),
                'cuda_available': torch.cuda.is_available(),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'error': str(e)
            }
            logger.error("Failed to move model to device", extra=error_context)
            raise RuntimeError(f"Failed to move model to {device}: {str(e)}") from e

    def create_pipeline(self) -> 'StableDiffusionXLPipeline':
        """
        Create a StableDiffusionXLPipeline for inference, ensuring all components are loaded.
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

        return StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer_1=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler
        )
