"""VAE encoder implementation with extreme speedups."""
import logging
from typing import Dict, Optional, Union
import torch
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)

class VAEEncoder:
    """Optimized VAE encoder wrapper for SDXL with extreme speedup."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16
    ):
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Apply optimizations
        self.vae.to(device=self.device, dtype=self.dtype, memory_format=torch.channels_last)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=False)
            
    def encode(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images to latent space with extreme optimization."""
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Invalid input tensor.")
            
        # Move to device and cast to correct dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            latents = self.vae.encode(pixel_values).latent_dist
            latents = latents.sample() * self.vae.config.scaling_factor
            return {"latent_dist": latents}
