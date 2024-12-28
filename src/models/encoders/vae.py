"""VAE encoder implementation with NaN protection."""
from typing import Dict, Optional, Union
import torch
from diffusers import AutoencoderKL
from src.core.logging import get_logger, LogConfig

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
        
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=False)
            
        logger.info(f"VAE encoder initialized on {device}")

    def _validate_tensor(self, tensor: torch.Tensor, name: str = "") -> torch.Tensor:
        """Validate and clean tensor values."""
        if tensor is None:
            raise ValueError(f"{name} tensor is None")
            
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            logger.warning(f"Found {nan_count} NaN values in {name}, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0)
            
        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            inf_count = inf_mask.sum().item()
            logger.warning(f"Found {inf_count} infinite values in {name}, clipping")
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
            
        return tensor.contiguous()

    def encode(
        self,
        pixel_values: torch.Tensor,
        num_images_per_prompt: int = 1,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode images to latent space.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            num_images_per_prompt: Number of samples per input
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with latent dist and unconditioned latents
        """
        try:
            # Input validation
            pixel_values = pixel_values.to(dtype=torch.float32)
            pixel_values = self._validate_tensor(pixel_values, "input")
            
            # Process with autocast and gradient disabled
            with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
                # Get encoding
                vae_output = self.vae.encode(pixel_values)
                
                # Sample from latent distribution
                if hasattr(vae_output, 'latent_dist'):
                    latents = vae_output.latent_dist.sample()
                else:
                    latents = vae_output.sample()
                    
                # Validate and scale latents
                latents = self._validate_tensor(latents, "latents")
                latents = latents * 0.18215

                # Generate unconditioned latents
                uncond_latents = torch.zeros_like(latents)
                
                # Handle multiple samples if requested
                if num_images_per_prompt > 1:
                    latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
                    uncond_latents = uncond_latents.repeat_interleave(num_images_per_prompt, dim=0)

                return {
                    "latent_dist": latents,
                    "uncond_latents": uncond_latents
                }

        except Exception as e:
            logger.error(f"VAE encoding failed: {e}")
            raise

    def encode_images(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images to latent space with validation.
        
        Args:
            pixel_values: Input tensor in [0, 1] range
            
        Returns:
            Dictionary with latents and metadata
        """
        try:
            # Input validation
            pixel_values = pixel_values.to(dtype=torch.float32)
            pixel_values = self._validate_tensor(pixel_values, "input")

            # Process image
            with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
                vae_output = self.vae.encode(pixel_values)
                
                # Get latents
                if hasattr(vae_output, 'latent_dist'):
                    latents = vae_output.latent_dist.sample()
                else:
                    latents = vae_output.sample()
                    
                # Validate and scale
                latents = self._validate_tensor(latents, "latents")
                latents = latents * 0.18215

                return {
                    "image_latent": latents,
                    "uncond_latents": torch.zeros_like(latents),
                    "metadata": {
                        "input_shape": tuple(pixel_values.shape),
                        "latent_shape": tuple(latents.shape),
                        "scaling_factor": 0.18215,
                        "stats": {
                            "min": latents.min().item(),
                            "max": latents.max().item(),
                            "mean": latents.mean().item(),
                            "std": latents.std().item()
                        }
                    }
                }

        except Exception as e:
            logger.error(f"VAE encoding failed: {e}")
            raise
