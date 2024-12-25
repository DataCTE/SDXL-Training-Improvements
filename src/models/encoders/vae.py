"""VAE encoder implementation with extreme speedups."""
from typing import Dict, Optional, Union
import torch
from diffusers import AutoencoderKL
from src.core.logging.logging import setup_logging

# Initialize logger with core logging system
logger = setup_logging(__name__)

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
                
        logger.info("VAE encoder initialized", extra={
            'device': str(self.device),
            'dtype': str(self.dtype),
            'model_type': type(self.vae).__name__
        })
            
    def encode(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images to latent space with extreme optimization."""
        try:
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Invalid input tensor.")
                
            logger.debug("Starting VAE encoding", extra={
                'input_shape': tuple(pixel_values.shape),
                'input_dtype': str(pixel_values.dtype),
                'input_device': str(pixel_values.device)
            })
                
            # Move to device and cast to correct dtype
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            latents = self.vae.encode(pixel_values).latent_dist
            latents = latents.sample() * self.vae.config.scaling_factor
            logger.debug("VAE encoding complete", extra={
                'output_shape': tuple(latents.shape),
                'output_dtype': str(latents.dtype),
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            })
            
            return {"latent_dist": latents}
                
        except Exception as e:
            logger.error("VAE encoding failed", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
                'stack_trace': True
            })
            raise
