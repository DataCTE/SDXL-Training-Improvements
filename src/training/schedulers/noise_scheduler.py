"""Noise scheduler implementation for SDXL training."""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from ...data.config import Config

logger = logging.getLogger(__name__)

@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise scheduler."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    clip_sample: bool = False
    set_alpha_to_one: bool = False
    steps_offset: int = 0
    prediction_type: str = "epsilon"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    rescale_betas_zero_snr: bool = False

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
    scheduler_config = NoiseSchedulerConfig(
        num_train_timesteps=config.model.num_timesteps,
        prediction_type=config.training.prediction_type,
        rescale_betas_zero_snr=config.training.zero_terminal_snr
    )
    
    scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config.num_train_timesteps,
        beta_start=scheduler_config.beta_start,
        beta_end=scheduler_config.beta_end,
        beta_schedule=scheduler_config.beta_schedule,
        clip_sample=scheduler_config.clip_sample,
        set_alpha_to_one=scheduler_config.set_alpha_to_one,
        steps_offset=scheduler_config.steps_offset,
        prediction_type=scheduler_config.prediction_type,
        thresholding=scheduler_config.thresholding,
        dynamic_thresholding_ratio=scheduler_config.dynamic_thresholding_ratio,
        sample_max_value=scheduler_config.sample_max_value,
        timestep_spacing=scheduler_config.timestep_spacing,
        rescale_betas_zero_snr=scheduler_config.rescale_betas_zero_snr
    )
    
    scheduler.to(device)
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
