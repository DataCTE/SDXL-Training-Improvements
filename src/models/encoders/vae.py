"""VAE encoder implementation with NaN protection."""
from typing import Dict, Optional, Union
import torch
from diffusers import AutoencoderKL
from src.core.logging import get_logger

logger = get_logger(__name__)

class VAEEncoder:
    """Optimized VAE encoder for stable diffusion."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16
    ):
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Force VAE to float32 for stability
        self.vae.to(device=self.device, dtype=torch.float32)
        
        logger.info(f"VAE encoder initialized on {device}")

    def encode_images(
        self,
        pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode images using VAE."""
        try:
            # Ensure input is on correct device and dtype
            if pixel_values.device != self.device:
                pixel_values = pixel_values.to(self.device)
            
            pixel_values = pixel_values.to(dtype=torch.float32)

            # Process image and keep on GPU
            with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
                model_input = self.vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * self.vae.config.scaling_factor

                return {
                    "model_input": model_input  # Keep on GPU
                }

        except Exception as e:
            logger.error(f"VAE encoding failed: {str(e)}")
            raise
