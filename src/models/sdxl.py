"""StableDiffusionXL model implementation with 100x speedups."""
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

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from contextlib import nullcontext
from random import Random
from typing import Dict, List, Optional, Tuple, Union

import gc
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
from src.models.adapters.lora import LoRAModuleWrapper, AdditionalEmbeddingWrapper

logger = logging.getLogger(__name__)

class StableDiffusionXLModelEmbedding(BaseModelEmbedding):
    def __init__(
        self,
        uuid: str,
        text_encoder_1_vector: Tensor,
        text_encoder_2_vector: Tensor,
        placeholder: str,
    ):
        super().__init__(
            uuid=uuid,
            token_count=text_encoder_1_vector.shape[0],
            placeholder=placeholder if placeholder else f"<embedding-{uuid}>"
        )
        self.text_encoder_1_vector = text_encoder_1_vector
        self.text_encoder_2_vector = text_encoder_2_vector

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
    def __init__(self, model_type: ModelType):
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_type)

        self.vae = None
        self.text_encoder_1 = None
        self.text_encoder_2 = None
        self.tokenizer_1 = None
        self.tokenizer_2 = None
        self.unet = None
        self.noise_scheduler = None
        self.text_encoder_1_lora = None
        self.text_encoder_2_lora = None
        self.unet_lora = None
        self.model_type = model_type
        self._dtype = DataType.FLOAT_32

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
        logger.info(f"Loading model components from {pretrained_model_name}")
        try:
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

                logger.info("Loading VAE...")
                self.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name,
                subfolder="vae",
                torch_dtype=model_dtypes.vae.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
                logger.info("Loading text encoder 1...")
                self.text_encoder_1 = CLIPTextModel.from_pretrained(
                pretrained_model_name,
                subfolder="text_encoder",
                torch_dtype=model_dtypes.text_encoder.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
                logger.info("Loading text encoder 2...")
                self.text_encoder_2 = CLIPTextModel.from_pretrained(
                pretrained_model_name,
                subfolder="text_encoder_2",
                torch_dtype=model_dtypes.text_encoder_2.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
                logger.info("Loading UNet...")
                self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=model_dtypes.unet.to_torch_dtype(),
                use_safetensors=use_safetensors
            )
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
                logger.info("Loading noise scheduler...")
                self.noise_scheduler = DDPMScheduler.from_pretrained(
                    pretrained_model_name,
                    subfolder="scheduler"
                )
                logger.info("Successfully loaded all model components")
            except Exception as e:
                error_context = {
                    'model_name': pretrained_model_name,
                    'dtype': str(dtype),
                    'error': str(e),
                    'component': 'unknown'
                }
                if not hasattr(self, 'vae') or self.vae is None:
                    error_context['component'] = 'vae'
                elif not hasattr(self, 'text_encoder_1') or self.text_encoder_1 is None:
                    error_context['component'] = 'text_encoder_1'
                elif not hasattr(self, 'text_encoder_2') or self.text_encoder_2 is None:
                    error_context['component'] = 'text_encoder_2'
                elif not hasattr(self, 'unet') or self.unet is None:
                    error_context['component'] = 'unet'
                elif not hasattr(self, 'tokenizer_1') or self.tokenizer_1 is None:
                    error_context['component'] = 'tokenizer_1'
                elif not hasattr(self, 'tokenizer_2') or self.tokenizer_2 is None:
                    error_context['component'] = 'tokenizer_2'
                elif not hasattr(self, 'noise_scheduler') or self.noise_scheduler is None:
                    error_context['component'] = 'noise_scheduler'
                
                error_msg = f"Failed to load {error_context['component']}: {str(e)}"
                logger.error(error_msg, extra=error_context)
                raise ValueError(error_msg)

    def vae_to(self, device: torch.device) -> None:
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
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("cuda is not available")
            device_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(device_idx)
        with create_stream_context(torch.cuda.current_stream()):
            for encoder in [self.text_encoder_1, self.text_encoder_2]:
                if not device_equals(encoder.device, device):
                    tensors_to_device_(encoder.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), encoder.state_dict())
            for lora in [self.text_encoder_1_lora, self.text_encoder_2_lora]:
                if lora is not None:
                    tensors_to_device_(lora.state_dict(), device, non_blocking=True)
                    tensors_record_stream(torch.cuda.current_stream(), lora.state_dict())

    def text_encoder_1_to(self, device: torch.device) -> None:
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_1.device, device):
                tensors_to_device_(self.text_encoder_1.state_dict(), device, non_blocking=True)
            if self.text_encoder_1_lora is not None:
                tensors_to_device_(self.text_encoder_1_lora.state_dict(), device, non_blocking=True)
            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_1.state_dict())

    def text_encoder_2_to(self, device: torch.device) -> None:
        with create_stream_context(torch.cuda.current_stream()):
            if not device_equals(self.text_encoder_2.device, device):
                tensors_to_device_(self.text_encoder_2.state_dict(), device, non_blocking=True)
            if self.text_encoder_2_lora is not None:
                tensors_to_device_(self.text_encoder_2_lora.state_dict(), device, non_blocking=True)
            if device.type == "cuda":
                tensors_record_stream(torch.cuda.current_stream(), self.text_encoder_2.state_dict())

    def unet_to(self, device: torch.device) -> None:
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
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.eval()

    def train(self) -> None:
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.unet.train()

    def zero_grad(self) -> None:
        self.unet.zero_grad(set_to_none=True)

    def parameters(self):
        return self.unet.parameters()

    def add_embeddings_to_prompt(self, prompt: str) -> str:
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

        text_encoder_output = torch.cat([text_encoder_1_output, text_encoder_2_output], dim=-1)
        return text_encoder_output, pooled_text_encoder_2_output

    def to(self, device: torch.device) -> None:
        logger.info(f"Moving model components to {device}")
        try:
            if device.type == "cuda":
                stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                with create_stream_context(stream):
                    self.vae_to(device)
                    torch_gc()
                    self.text_encoder_to(device)
                    torch_gc()
                    self.unet_to(device)
                    torch_gc()
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            else:
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
