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
        """Encode images to latent space with enhanced validation and diagnostics."""
        try:
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Invalid input tensor.")
                
            # Log input tensor stats
            logger.info("VAE input tensor stats:", extra={
                'input_shape': tuple(pixel_values.shape),
                'input_dtype': str(pixel_values.dtype),
                'input_device': str(pixel_values.device),
                'input_min': pixel_values.min().item(),
                'input_max': pixel_values.max().item(),
                'input_mean': pixel_values.mean().item(),
                'input_std': pixel_values.std().item()
            })

            # Validate input tensor
            if torch.isnan(pixel_values).any():
                nan_count = torch.isnan(pixel_values).sum().item()
                raise ValueError(f"Input tensor contains {nan_count} NaN values")
            if torch.isinf(pixel_values).any():
                inf_count = torch.isinf(pixel_values).sum().item()
                raise ValueError(f"Input tensor contains {inf_count} infinite values")
            if not (0 <= pixel_values.min() <= pixel_values.max() <= 1.0):
                raise ValueError(
                    f"Input tensor values out of range [0,1]: "
                    f"min={pixel_values.min().item()}, max={pixel_values.max().item()}"
                )
                
            # Move to device and cast to correct dtype
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            
            # Use context managers inside try block
            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
                # Log intermediate values through VAE encoding stages
                logger.debug("Starting VAE encode pass")
                
                # Get initial distribution
                dist = self.vae.encode(pixel_values)
                logger.debug("VAE encoder output stats:", extra={
                    'dist_mean': dist.mean.mean().item(),
                    'dist_std': dist.std.mean().item() if hasattr(dist, 'std') else None
                })
                
                # Sample from distribution
                latents = dist.latent_dist
                logger.debug("Pre-scaling latent stats:", extra={
                    'latent_shape': tuple(latents.shape),
                    'latent_min': latents.min().item(),
                    'latent_max': latents.max().item(),
                    'latent_mean': latents.mean().item(),
                    'latent_std': latents.std().item()
                })
                
                # Sample and scale
                latents = latents.sample()
                logger.debug("Post-sample latent stats:", extra={
                    'sampled_min': latents.min().item(),
                    'sampled_max': latents.max().item(),
                    'sampled_mean': latents.mean().item(),
                    'sampled_std': latents.std().item()
                })
                
                # Apply scaling factor
                scaling_factor = self.vae.config.scaling_factor
                latents = latents * scaling_factor
                
                # Final validation
                if torch.isnan(latents).any():
                    nan_locations = torch.isnan(latents)
                    nan_count = nan_locations.sum().item()
                    nan_indices = torch.where(nan_locations)
                    raise ValueError(
                        f"VAE produced {nan_count} NaN values in latents. "
                        f"First NaN at index: {[idx[0].item() for idx in nan_indices]}"
                    )
                    
                if torch.isinf(latents).any():
                    inf_count = torch.isinf(latents).sum().item()
                    raise ValueError(f"VAE produced {inf_count} infinite values in latents")

                logger.info("VAE encoding complete", extra={
                    'output_shape': tuple(latents.shape),
                    'output_dtype': str(latents.dtype),
                    'scaling_factor': scaling_factor,
                    'output_stats': {
                        'min': latents.min().item(),
                        'max': latents.max().item(),
                        'mean': latents.mean().item(),
                        'std': latents.std().item()
                    }
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
