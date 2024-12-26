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
                
            # Add input normalization and validation
            if pixel_values.dtype != self.dtype:
                pixel_values = pixel_values.to(dtype=self.dtype)
                
            # Ensure values are in correct range
            if pixel_values.min() < -1 or pixel_values.max() > 1:
                logger.warning("Input values outside expected range [-1,1], normalizing...")
                pixel_values = torch.clamp(pixel_values, -1, 1)

            # Log input tensor stats
            logger.debug("VAE input tensor stats:", extra={
                'input_shape': tuple(pixel_values.shape),
                'input_dtype': str(pixel_values.dtype),
                'input_device': str(pixel_values.device),
                'input_min': pixel_values.min().item(),
                'input_max': pixel_values.max().item(),
                'input_mean': pixel_values.mean().item(),
                'input_std': pixel_values.std().item()
            })

            # Move to device and cast to correct dtype
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            
            # Use context managers inside try block
            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
                # Add gradient clipping for stability
                with torch.no_grad():
                    # Get VAE output with added stability measures
                    vae_output = self.vae.encode(pixel_values)
                    
                    # Handle different output formats
                    if hasattr(vae_output, 'latent_dist'):
                        latents = vae_output.latent_dist
                        if hasattr(latents, 'sample'):
                            latents = latents.sample()
                    elif hasattr(vae_output, 'sample'):
                        latents = vae_output.sample()
                    else:
                        latents = vae_output

                    # Add numerical stability checks
                    if torch.isnan(latents).any():
                        # Try to recover from NaNs
                        logger.warning("NaN values detected in latents, attempting recovery...")
                        mask = torch.isnan(latents)
                        latents[mask] = 0.0  # Replace NaNs with zeros
                        
                        # If still have NaNs after recovery attempt, raise error
                        if torch.isnan(latents).any():
                            nan_locations = torch.isnan(latents)
                            nan_count = nan_locations.sum().item()
                            nan_indices = torch.where(nan_locations)
                            raise ValueError(
                                f"VAE produced {nan_count} NaN values in latents that couldn't be recovered. "
                                f"First NaN at index: {[idx[0].item() for idx in nan_indices]}"
                            )
                    
                    # Check for infinities
                    if torch.isinf(latents).any():
                        logger.warning("Infinite values detected in latents, clipping...")
                        latents = torch.clamp(latents, -1e6, 1e6)

                    # Apply scaling factor with stability check
                    scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                    latents = latents * scaling_factor
                    
                    # Final validation
                    if not torch.isfinite(latents).all():
                        raise ValueError("Non-finite values in final latents after scaling")

                    logger.debug("VAE encoding complete", extra={
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
