"""StableDiffusionXL model implementation with optimized encoders."""
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline as BasePipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from src.core.logging import get_logger
from src.models.encoders import CLIPEncoder, VAEEncoder

logger = get_logger(__name__)

class StableDiffusionXL:
    """Optimized SDXL implementation with improved encoders."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder_1: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_1: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16,
    ):
        """Initialize SDXL model with optimized encoders."""
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Initialize optimized encoders
        self.vae_encoder = VAEEncoder(
            vae=vae,
            device=device,
            dtype=dtype
        )
        
        self.text_encoders = [text_encoder_1, text_encoder_2]
        self.tokenizers = [tokenizer_1, tokenizer_2]
        
        # Store other components
        self.unet = unet.to(device=device, dtype=dtype)
        self.scheduler = scheduler
        
        logger.info(f"Initialized SDXL model on {device}")

    def generate_timestep_weights(
        self,
        num_timesteps: int,
        bias_strategy: str = "none",
        bias_portion: float = 0.25,
        bias_multiplier: float = 2.0,
        bias_begin: Optional[int] = None,
        bias_end: Optional[int] = None
    ) -> torch.Tensor:
        """Generate weighted timestep sampling distribution.
        
        Args:
            num_timesteps: Total number of timesteps
            bias_strategy: Strategy for biasing timesteps ('none', 'earlier', 'later', 'range')
            bias_portion: Portion of timesteps to bias when using earlier/later strategies
            bias_multiplier: Weight multiplier for biased timesteps
            bias_begin: Starting timestep for range strategy
            bias_end: Ending timestep for range strategy
            
        Returns:
            Tensor of timestep weights normalized to sum to 1
        """
        weights = torch.ones(num_timesteps, device=self.device)
        
        # Early return if no bias
        if bias_strategy == "none":
            return weights / weights.sum()
            
        if bias_multiplier <= 0:
            raise ValueError(
                "Timestep bias multiplier must be positive. "
                "Use bias_strategy='none' to disable timestep biasing."
            )

        # Calculate number of timesteps to bias
        num_to_bias = int(bias_portion * num_timesteps)

        # Apply bias based on strategy
        if bias_strategy == "later":
            weights[-num_to_bias:] *= bias_multiplier
        elif bias_strategy == "earlier":
            weights[:num_to_bias] *= bias_multiplier
        elif bias_strategy == "range":
            if bias_begin is None or bias_end is None:
                raise ValueError("bias_begin and bias_end must be specified for range strategy")
            if bias_begin < 0 or bias_end > num_timesteps:
                raise ValueError(
                    f"Bias range must be within [0, {num_timesteps}], "
                    f"got [{bias_begin}, {bias_end}]"
                )
            weights[bias_begin:bias_end] *= bias_multiplier
        else:
            raise ValueError(
                f"Unknown bias strategy: {bias_strategy}. "
                "Must be one of: none, earlier, later, range"
            )

        # Normalize weights
        return weights / weights.sum()

    def encode_prompt(
        self,
        batch: Dict[str, Any],
        proportion_empty_prompts: float = 0.0,
        caption_column: str = "prompt",
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts using both CLIP encoders."""
        return CLIPEncoder.encode_prompt(
            batch=batch,
            text_encoders=self.text_encoders,
            tokenizers=self.tokenizers,
            proportion_empty_prompts=proportion_empty_prompts,
            caption_column=caption_column,
            is_train=is_train
        )

    def encode_images(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode images using VAE."""
        return self.vae_encoder.encode_images(
            pixel_values=pixel_values
        )

    def create_pipeline(self) -> BasePipeline:
        """Create a StableDiffusionXLPipeline for inference."""
        return BasePipeline(
            vae=self.vae_encoder.vae,
            text_encoder=self.text_encoders[0],
            text_encoder_2=self.text_encoders[1], 
            tokenizer=self.tokenizers[0],
            tokenizer_2=self.tokenizers[1],
            unet=self.unet,
            scheduler=self.scheduler
        )
