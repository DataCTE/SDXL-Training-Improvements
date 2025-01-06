"""Noise scheduler implementation for SDXL training with ZTSNR support."""
import logging
from typing import List, Optional, Tuple, Union, Callable

import torch
from diffusers import DDPMScheduler
from src.data.config import Config

logger = logging.getLogger(__name__)

class NoiseScheduler:
    """Enhanced noise scheduler with ZTSNR and v-prediction support."""
    
    def __init__(self, config: Config, device: Union[str, torch.device]):
        self.config = config
        self.device = device
        self.sigma_data = 1.0  # Standard for latent diffusion
        
        # Initialize base scheduler
        self.base_scheduler = DDPMScheduler(
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
        self._initialize_scheduler(device)

    def _initialize_scheduler(self, device: Union[str, torch.device]) -> None:
        """Initialize scheduler and move tensors to device."""
        if not hasattr(self.base_scheduler, 'alphas_cumprod'):
            logger.warning("Scheduler missing alphas_cumprod - initializing")
            self.base_scheduler.set_timesteps(self.base_scheduler.num_train_timesteps)

        # Move tensors to device
        tensor_attributes = [
            'betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
            'posterior_log_variance_clipped', 'posterior_mean_coef1',
            'posterior_mean_coef2', 'final_alpha_cumprod'
        ]
        
        for attr in tensor_attributes:
            if hasattr(self.base_scheduler, attr):
                tensor = getattr(self.base_scheduler, attr)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    setattr(self.base_scheduler, attr, tensor.to(device))
                    logger.debug(f"Moved scheduler.{attr} to {device}")

    def get_karras_scalings(self, sigma: torch.Tensor, sigma_data: float = 1.0):
        """Compute Karras scalings for v-prediction."""
        c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
        c_out = -sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        return c_skip, c_out, c_in

    def get_infinite_karras_scalings(self, sigma_data: float = 1.0):
        """Compute simplified Karras scalings when σ approaches infinity."""
        c_skip = 0  # As σ² dominates denominator
        c_out = -sigma_data  # Limit as σ approaches infinity
        return c_skip, c_out

    def ztsnr_first_step(self, n: torch.Tensor, sigma_1: float, model_fn: Callable) -> torch.Tensor:
        """Execute first sampling step with ZTSNR."""
        x1 = sigma_1 * n - self.sigma_data * model_fn(n, torch.tensor([float('inf')]))
        return x1

    def euler_step(self, x: torch.Tensor, sigma_i: float, sigma_next: float, 
                  model_fn: Callable) -> torch.Tensor:
        """Regular Euler step for remaining sampling steps."""
        c_skip, c_out, c_in = self.get_karras_scalings(sigma_i)
        denoised = c_skip * x + c_out * model_fn(c_in * x, sigma_i)
        d = (x - denoised) / sigma_i
        x_next = x + (sigma_next - sigma_i) * d
        return x_next

    def sample_with_ztsnr(self, model_fn: Callable, latent_shape: Tuple[int, ...], 
                         num_steps: int) -> torch.Tensor:
        """Sample using ZTSNR for first step."""
        n = torch.randn(latent_shape, device=self.device)
        sigmas = self.get_sigmas(num_steps)
        
        # Special first step using ZTSNR
        x = self.ztsnr_first_step(n, sigmas[0], model_fn)
        
        # Continue with regular sampling
        for i in range(1, len(sigmas)):
            x = self.euler_step(x, sigmas[i-1], sigmas[i], model_fn)
        
        return x

    def get_sigmas(self, n_sigmas: int) -> torch.Tensor:
        """Get noise schedule sigma values."""
        return get_karras_sigmas(
            n_sigmas=n_sigmas,
            sigma_min=self.config.model.sigma_min,
            sigma_max=self.config.model.sigma_max if not self.config.model.use_ztsnr else 20000.0,
            rho=self.config.model.rho,
            device=self.device
        )

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, 
                 timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples with ZTSNR support."""
        sigmas = self.timestep_to_sigma(timesteps)
        noisy = sample + sigmas.view(-1, 1, 1, 1) * noise
        
        if self.config.model.use_ztsnr:
            noisy = torch.clamp(noisy, -20000.0, 20000.0)
            
        return noisy

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, 
                    timesteps: torch.Tensor) -> torch.Tensor:
        """Compute velocity for v-prediction."""
        sigmas = self.timestep_to_sigma(timesteps)
        velocity = (noise - sample) / (sigmas.view(-1, 1, 1, 1) ** 2).sqrt()
        return velocity

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute SNR for MinSNR loss weighting."""
        sigmas = self.timestep_to_sigma(timesteps)
        return (self.sigma_data / sigmas) ** 2

    def timestep_to_sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to noise levels."""
        sigmas = self.get_sigmas(self.config.model.num_timesteps)
        return sigmas[timesteps]

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps with proper ZTSNR distribution."""
        if self.config.model.use_ztsnr:
            # Log-uniform sampling for ZTSNR
            u = torch.rand(batch_size, device=self.device)
            timesteps = (u * self.config.model.num_timesteps).long()
        else:
            # Regular uniform sampling
            timesteps = torch.randint(
                0, self.config.model.num_timesteps,
                (batch_size,), device=self.device
            )
        return timesteps

def configure_noise_scheduler(
    config: Config,
    device: Union[str, torch.device]
) -> NoiseScheduler:
    """Configure enhanced noise scheduler."""
    return NoiseScheduler(config, device)

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
