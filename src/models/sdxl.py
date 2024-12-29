"""StableDiffusionXL model implementation with optimized encoders."""
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from diffusers import StableDiffusionXLPipeline
from src.core.logging import get_logger
from src.models.encoders import CLIPEncoder, VAEEncoder
from src.models.base import ModelType

logger = get_logger(__name__)

class StableDiffusionXL:
    """SDXL model wrapper with memory optimizations."""
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        device: Optional[torch.device] = None,
        model_type: ModelType = ModelType.SDXL,
        **kwargs: Any
    ) -> "StableDiffusionXL":
        """Load pretrained SDXL model."""
        logger.info(f"Loading {model_type.value} model from {pretrained_model_name}")
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            **kwargs
        )
        
        # Move to device if specified
        if device is not None:
            pipeline = pipeline.to(device)
            
        # Create instance
        instance = cls()
        instance.pipeline = pipeline
        instance.unet = pipeline.unet
        instance.model_type = model_type
        
        # Initialize optimized encoders
        instance.vae_encoder = VAEEncoder(
            vae=pipeline.vae,
            device=device,
            dtype=torch.float16
        )
        
        instance.text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
        instance.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
        
        instance.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add direct access to components for compatibility
        instance.vae = instance.vae_encoder.vae
        instance.text_encoder_1 = instance.text_encoders[0]
        instance.text_encoder_2 = instance.text_encoders[1]
        instance.tokenizer_1 = instance.tokenizers[0]
        instance.tokenizer_2 = instance.tokenizers[1]
        instance.scheduler = pipeline.scheduler
        
        logger.info(f"Initialized {model_type.value} model on {device}")
        return instance

    def to(self, device: torch.device) -> "StableDiffusionXL":
        """Move model to device."""
        self.pipeline = self.pipeline.to(device)
        self.unet = self.unet.to(device)
        self.vae_encoder = self.vae_encoder.to(device)
        self.text_encoders = [encoder.to(device) for encoder in self.text_encoders]
        self.device = device
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Get model state dict."""
        return {
            "unet": self.unet.state_dict(),
            "vae": self.vae_encoder.vae.state_dict(),
            "text_encoder": self.text_encoders[0].state_dict(),
            "text_encoder_2": self.text_encoders[1].state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict."""
        self.unet.load_state_dict(state_dict["unet"])
        self.vae_encoder.vae.load_state_dict(state_dict["vae"])
        self.text_encoders[0].load_state_dict(state_dict["text_encoder"])
        self.text_encoders[1].load_state_dict(state_dict["text_encoder_2"])

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

    def create_pipeline(self) -> StableDiffusionXLPipeline:
        """Create a StableDiffusionXLPipeline for inference."""
        return StableDiffusionXLPipeline(
            vae=self.vae_encoder.vae,
            text_encoder=self.text_encoders[0],
            text_encoder_2=self.text_encoders[1], 
            tokenizer=self.tokenizers[0],
            tokenizer_2=self.tokenizers[1],
            unet=self.unet,
            scheduler=self.scheduler
        )
