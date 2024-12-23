"""VAE encoder implementation with optimized memory handling."""
import logging
from typing import Dict, Optional, Tuple, Union
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from src.core.memory.tensor import (
    tensors_to_device_,
    create_stream_context,
    tensors_record_stream,
    torch_sync,
    pin_tensor_,
    unpin_tensor_,
    device_equals
)
from src.core.types import DataType

logger = logging.getLogger(__name__)

class VAEEncoder:
    """Optimized VAE encoder wrapper for SDXL."""
    
    def __init__(
        self,
        vae: AutoencoderKL,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
        enable_memory_efficient_attention: bool = True,
        enable_vae_slicing: bool = False,
        enable_gradient_checkpointing: bool = True
    ):
        """Initialize VAE encoder with performance optimizations."""
        """Initialize VAE encoder with memory optimizations.
        
        Args:
            vae: Base VAE model
            device: Target device
            dtype: Model dtype
            enable_memory_efficient_attention: Use memory efficient attention
            enable_vae_slicing: Enable VAE slicing for large images
            enable_vae_tiling: Enable VAE tiling
            vae_tile_size: Tile size for VAE tiling
            enable_gradient_checkpointing: Enable gradient checkpointing
        """
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype or DataType.FLOAT_32.torch_dtype()
        
        # Configure memory optimizations
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_vae_slicing = enable_vae_slicing
        
        # Setup model with optimizations
        self.vae.to(device=self.device, dtype=self.dtype, memory_format=torch.channels_last)
        if enable_gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
            
        # Enable memory efficient attention
        if hasattr(self.vae, "set_use_memory_efficient_attention_xformers"):
            self.vae.set_use_memory_efficient_attention_xformers(True)
            
        # Compile model for faster execution if torch.compile is available
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=True)
            
        # Track memory stats
        self.peak_memory = 0
        self.current_memory = 0
        
    def encode(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Encode images to latent space with memory optimization.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Whether to return dict with distribution params
            
        Returns:
            Encoded latents or dict with distribution parameters
        """
        try:
            # Track initial memory
            if torch.cuda.is_available():
                self.current_memory = torch.cuda.memory_allocated()
                
            # Validate input
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(pixel_values)}")
                
            if pixel_values.dim() != 4:
                raise ValueError(f"Expected 4D tensor, got {pixel_values.dim()}D")
                
            # Setup processing streams
            compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
            
            with create_stream_context(compute_stream):
                # Move to device if needed
                if not device_equals(pixel_values.device, self.device):
                    tensors_to_device_(pixel_values, self.device)
                    
                # Pin memory
                pin_tensor_(pixel_values)
                
                try:
                    # Encode with VAE
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist
                            
                    # Scale latents
                    latents = latents.sample() * self.vae.config.scaling_factor
                    
                    # Record stream
                    if compute_stream:
                        tensors_record_stream(compute_stream, latents)
                        
                    # Update peak memory
                    if torch.cuda.is_available():
                        current = torch.cuda.memory_allocated()
                        self.peak_memory = max(self.peak_memory, current)
                        self.current_memory = current
                        
                    return {"latent_dist": latents} if return_dict else latents
                    
                finally:
                    # Cleanup
                    unpin_tensor_(pixel_values)
                    torch_sync()
                    
        except Exception as e:
            logger.error(f"VAE encoding failed: {str(e)}")
            raise
            
        
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory usage statistics.
        
        Returns:
            Dict with current and peak memory usage
        """
        return {
            "current_memory": self.current_memory,
            "peak_memory": self.peak_memory
        }
