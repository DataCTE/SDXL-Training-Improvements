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

    def encode(
         self,
         pixel_values: torch.Tensor,
         num_images_per_prompt: int = 1,
         output_hidden_states: bool = False
     ) -> Dict[str, torch.Tensor]:
         """Encode images to latent space with enhanced validation and diffusers-style processing.    

         Args:
             pixel_values: Input image tensor
             num_images_per_prompt: Number of images to generate per prompt
             output_hidden_states: Whether to output hidden states instead of embeddings
 
         Returns:
             Dictionary containing encoded latents or hidden states
         """
         try:
             # Input validation and normalization
             if not isinstance(pixel_values, torch.Tensor):
                 raise ValueError("Input must be a tensor")

             # Ensure input is in correct range [0, 1]
             if pixel_values.min() < -0.1 or pixel_values.max() > 1.1:
                 logger.warning(
                     f"Input tensor out of range: min={pixel_values.min().item():.3f}, "
                     f"max={pixel_values.max().item():.3f}"
                 )
                 pixel_values = torch.clamp(pixel_values, 0, 1)

             # Normalize to [-1, 1] range expected by VAE
             pixel_values = 2 * pixel_values - 1

             # Get VAE dtype
             dtype = next(self.vae.parameters()).dtype

             # Convert input to correct dtype and device
             pixel_values = pixel_values.to(device=self.device, dtype=dtype)

             # Log input tensor stats
             logger.debug("VAE input tensor stats:", extra={
                 'input_shape': tuple(pixel_values.shape),
                 'input_dtype': str(pixel_values.dtype),
                 'input_device': str(pixel_values.device),
                 'input_range': {
                     'min': pixel_values.min().item(),
                     'max': pixel_values.max().item(),
                     'mean': pixel_values.mean().item(),
                     'std': pixel_values.std().item()
                 }
             })

             # Process with error handling and gradient disabled
             with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):       
                 if output_hidden_states:
                     # Get hidden states
                     vae_output = self.vae(pixel_values, output_hidden_states=True)
                     hidden_states = vae_output.hidden_states[-2]

                     # Generate unconditioned hidden states
                     uncond_output = self.vae(
                         torch.zeros_like(pixel_values),
                         output_hidden_states=True
                     )
                     uncond_hidden_states = uncond_output.hidden_states[-2]

                     # Repeat for each requested image
                     hidden_states = hidden_states.repeat_interleave(num_images_per_prompt, dim=0)    
                     uncond_hidden_states = uncond_hidden_states.repeat_interleave(
                         num_images_per_prompt, dim=0
                     )

                     # Validate outputs
                     for tensor, name in [(hidden_states, "hidden_states"),
                                        (uncond_hidden_states, "uncond_hidden_states")]:
                         if torch.isnan(tensor).any():
                             nan_count = torch.isnan(tensor).sum().item()
                             nan_indices = torch.where(torch.isnan(tensor))
                             raise ValueError(
                                 f"VAE produced {nan_count} NaN values in {name}. "
                                 f"First NaN at index: {[idx[0].item() for idx in nan_indices]}"      
                             )
                         if torch.isinf(tensor).any():
                             tensor = torch.clamp(tensor, -1e6, 1e6)

                     return {
                         "hidden_states": hidden_states,
                         "uncond_hidden_states": uncond_hidden_states
                     }

                 else:
                     # Get latent distribution
                     vae_output = self.vae.encode(pixel_values)

                     if hasattr(vae_output, 'latent_dist'):
                         # Sample from distribution with reduced variance
                         latents = vae_output.latent_dist.sample() * 0.18215
                     else:
                         latents = vae_output.sample() * 0.18215

                     # Generate unconditioned latents
                     uncond_latents = torch.zeros_like(latents)

                     # Repeat for each requested image
                     latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
                     uncond_latents = uncond_latents.repeat_interleave(num_images_per_prompt, dim=0)  

                     # Validate outputs
                     for tensor, name in [(latents, "latents"), (uncond_latents, "uncond_latents")]:  
                         if torch.isnan(tensor).any():
                             # Try to recover from NaNs by clamping and re-normalizing
                             logger.warning(f"NaN values detected in {name}, attempting recovery...") 
                             tensor = torch.nan_to_num(tensor, nan=0.0)
                             tensor = torch.clamp(tensor, -1e6, 1e6)
                             if torch.isnan(tensor).any():
                                 nan_count = torch.isnan(tensor).sum().item()
                                 nan_indices = torch.where(torch.isnan(tensor))
                                 raise ValueError(
                                     f"VAE produced {nan_count} NaN values in {name}. "
                                     f"First NaN at index: {[idx[0].item() for idx in nan_indices]}"  
                                 )
                         if torch.isinf(tensor).any():
                             tensor = torch.clamp(tensor, -1e6, 1e6)

                     logger.debug("VAE encoding complete", extra={
                         'output_shape': tuple(latents.shape),
                         'output_dtype': str(latents.dtype),
                         'scaling_factor': 0.18215,
                         'output_stats': {
                             'min': latents.min().item(),
                             'max': latents.max().item(),
                             'mean': latents.mean().item(),
                             'std': latents.std().item()
                         }
                     })

                     return {
                         "latent_dist": latents,
                         "uncond_latents": uncond_latents
                     }

         except Exception as e:
             logger.error("VAE encoding failed", extra={
                 'error_type': type(e).__name__,
                 'error': str(e),
                 'input_shape': tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor)   
 else None,
                 'stack_trace': True
             })
             raise
