"""Noise generation and processing utilities for SDXL training."""
import logging
import torch
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)

def generate_noise(
    shape: Tuple[int, ...],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    generator: torch.Generator = None,
    layout: torch.Tensor = None
) -> torch.Tensor:
    """Generate noise tensor for latent diffusion.
    
    Args:
        shape: Shape of noise tensor
        device: Target device
        dtype: Target dtype
        generator: Optional random generator
        layout: Optional tensor to match memory layout
        
    Returns:
        Noise tensor
    """
    try:
        if layout is not None:
            # Match memory layout of input
            noise = torch.randn(
                shape,
                generator=generator,
                device='cpu',
                dtype=dtype
            )
            noise = noise.to(device=device, memory_format=layout.memory_format)
        else:
            # Generate directly on target device
            noise = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype
            )
        return noise
        
    except Exception as e:
        logger.error(f"Error generating noise: {str(e)}")
        raise

def get_add_time_ids(
    original_sizes: List[Tuple[int, int]],
    crops_coords_top_lefts: List[Tuple[int, int]],
    target_sizes: List[Tuple[int, int]],
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> torch.Tensor:
    """Get added time IDs for SDXL conditioning.
    
    Args:
        original_sizes: List of original image sizes (H,W)
        crops_coords_top_lefts: List of crop coordinates (top,left)
        target_sizes: List of target sizes (H,W)
        dtype: Target dtype
        device: Target device
        
    Returns:
        Tensor of time IDs
    """
    try:
        add_time_ids = []

        for original_size, crop_coords_top_left, target_size in zip(
            original_sizes, crops_coords_top_lefts, target_sizes
        ):
            # Original size
            add_time_ids.append([original_size[0], original_size[1]])
            
            # Crop coordinates
            add_time_ids.append([crop_coords_top_left[0], crop_coords_top_left[1]])
            
            # Target size
            add_time_ids.append([target_size[0], target_size[1]])

        add_time_ids = torch.tensor(add_time_ids, dtype=dtype, device=device)
        add_time_ids = add_time_ids.reshape(len(original_sizes), 6)
        
        return add_time_ids
        
    except Exception as e:
        logger.error(f"Error generating time IDs: {str(e)}")
        raise
