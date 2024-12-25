"""VAE encoder implementation with extreme speedups."""
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

# Force speed optimizations
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

class VAEEncoder:
    """Optimized VAE encoder wrapper for SDXL with extreme speedup."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16
    ):
        """Initialize VAE encoder with extreme performance optimizations.
        
        Args:
            vae: Base VAE model
            device: Target device
            dtype: Model dtype (defaults to float16 for speed)
        """
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Apply aggressive optimizations
        self.vae.to(device=self.device, dtype=self.dtype, memory_format=torch.channels_last)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=False)
            
        # Track memory stats
        self.peak_memory = 0
        self.current_memory = 0
        
    def encode(self, pixel_values: torch.Tensor, return_dict: bool = True):
        """Encode images to latent space with extreme optimization.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return dict with distribution params
            
        Returns:
            Encoded latents or dict with distribution parameters
        """
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.dim() != 4:
            raise ValueError("Invalid input tensor.")
            
        # Move to device and cast to correct dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        
        # Use inference mode and autocast for maximum speed
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            latents = self.vae.encode(pixel_values).latent_dist
            latents = latents.sample() * self.vae.config.scaling_factor
            
            # Track memory usage
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated()
                self.peak_memory = max(self.peak_memory, current)
                self.current_memory = current
                
            return {"latent_dist": latents} if return_dict else latents
            
        
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return {
            "current_memory": self.current_memory,
            "peak_memory": self.peak_memory
        }