"""Noise scheduler implementation for SDXL training."""
import logging
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DDPMScheduler
from src.data.config import Config

logger = logging.getLogger(__name__)

def configure_noise_scheduler(
    config: Config,
    device: Union[str, torch.device]
) -> DDPMScheduler:
    """Configure noise scheduler for training.
    
    Args:
        config: Training configuration
        device: Target device
        
    Returns:
        Configured noise scheduler
    """
    # Create scheduler with device-aware tensors
    scheduler = DDPMScheduler(
        num_train_timesteps=config.training.method_config.scheduler.num_train_timesteps,
        beta_start=config.training.method_config.scheduler.beta_start,
        beta_end=config.training.method_config.scheduler.beta_end,
        beta_schedule=config.training.method_config.scheduler.beta_schedule,
        clip_sample=True,
        steps_offset=config.training.method_config.scheduler.steps_offset,
        prediction_type=config.model.prediction_type,
        thresholding=True,
        dynamic_thresholding_ratio=config.training.method_config.scheduler.dynamic_thresholding_ratio,
        sample_max_value=20000.0,
        timestep_spacing=config.training.method_config.scheduler.timestep_spacing,
        rescale_betas_zero_snr=True
    )
    
    # Ensure scheduler is initialized
    if not hasattr(scheduler, 'alphas_cumprod'):
        logger.warning("Scheduler missing alphas_cumprod - initializing")
        scheduler.set_timesteps(scheduler.num_train_timesteps)
    
    # Move all relevant tensors to device
    tensor_attributes = [
        'betas',
        'alphas',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_alphas_cumprod',
        'sqrt_one_minus_alphas_cumprod', 
        'log_one_minus_alphas_cumprod',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'posterior_variance',
        'posterior_log_variance_clipped',
        'posterior_mean_coef1',
        'posterior_mean_coef2',
        'final_alpha_cumprod'
    ]
    
    for attr in tensor_attributes:
        if hasattr(scheduler, attr):
            tensor = getattr(scheduler, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(scheduler, attr, tensor.to(device))
                logger.debug(f"Moved scheduler.{attr} to {device}")
            elif tensor is None:
                logger.warning(f"Scheduler.{attr} is None")
    
    # Verify critical tensors
    if scheduler.alphas_cumprod is None:
        raise ValueError("Scheduler alphas_cumprod is None after initialization")
        
    logger.debug(f"Scheduler initialized with {len(scheduler.alphas_cumprod)} timesteps")
    
    return scheduler

def get_karras_sigmas(
    n_sigmas: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Get sigmas for Karras noise schedule.
    
    Args:
        n_sigmas: Number of sigma values
        sigma_min: Minimum sigma
        sigma_max: Maximum sigma
        rho: Schedule parameter
        device: Target device
        
    Returns:
        Tensor of sigma values
    """
    ramp = torch.linspace(0, 1, n_sigmas, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    return sigmas

def get_sigmas(
    config: Config,
    n_sigmas: int,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """Get noise schedule sigma values.
    
    Args:
        config: Training configuration
        n_sigmas: Number of sigma values
        device: Target device
        
    Returns:
        Tensor of sigma values
    """
    return get_karras_sigmas(
        n_sigmas=n_sigmas,
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        rho=config.model.rho,
        device=device
    )

def get_add_time_ids(
    original_sizes: List[Tuple[int, int]],
    crop_top_lefts: List[Tuple[int, int]],
    target_sizes: List[Tuple[int, int]],
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> torch.Tensor:
    """Get time embeddings for SDXL conditioning.
    
    Args:
        original_sizes: Original image sizes
        crop_top_lefts: Crop coordinates
        target_sizes: Target sizes after transforms
        dtype: Tensor dtype
        device: Target device
        
    Returns:
        Time embedding tensor
    """
    add_time_ids = [
        list(original_size) + list(crop_top_left) + list(target_size)
        for original_size, crop_top_left, target_size 
        in zip(original_sizes, crop_top_lefts, target_sizes)
    ]
    
    add_time_ids = torch.tensor(add_time_ids, dtype=dtype, device=device)
    return add_time_ids

def get_scheduler_parameters(
    config: Config,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get noise scheduler parameters.
    
    Args:
        config: Training configuration
        device: Target device
        
    Returns:
        Tuple of (sigmas, timesteps)
    """
    sigmas = get_sigmas(
        config=config,
        n_sigmas=config.model.num_timesteps,
        device=device
    )
    timesteps = torch.arange(
        len(sigmas) - 1, -1, -1, 
        dtype=torch.long,
        device=device
    )
    
    return sigmas, timesteps
