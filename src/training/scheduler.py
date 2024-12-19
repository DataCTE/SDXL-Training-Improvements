"""Noise scheduler and related functions for SDXL training."""
import logging
import torch
from typing import Dict, Any, Tuple
from ..data.config import Config

logger = logging.getLogger(__name__)

def get_karras_scalings(sigmas: torch.Tensor, timestep_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Karras noise schedule scalings for given timesteps."""
    try:
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError(f"Expected sigmas to be torch.Tensor, got {type(sigmas)}")
        if not isinstance(timestep_indices, torch.Tensor):
            raise ValueError(f"Expected timestep_indices to be torch.Tensor, got {type(timestep_indices)}")
            
        timestep_sigmas = sigmas[timestep_indices]
        c_skip = 1 / (timestep_sigmas**2 + 1).sqrt()
        c_out = -timestep_sigmas / (timestep_sigmas**2 + 1).sqrt()
        c_in = 1 / (timestep_sigmas**2 + 1).sqrt()
        return c_skip, c_out, c_in
        
    except Exception as e:
        logger.error(f"Error in get_karras_scalings: {str(e)}")
        raise

def get_sigmas(config: "Config", device: torch.device) -> torch.Tensor:
    """Generate noise schedule for ZTSNR with optimized scaling."""
    try:
        if not isinstance(config, Config):
            raise ValueError(f"Expected config to be Config, got {type(config)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"Expected device to be torch.device, got {type(device)}")
            
        num_timesteps = config.model.num_timesteps
        sigma_min = config.model.sigma_min
        sigma_max = config.model.sigma_max
        rho = config.model.rho
        
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}")
        
        # Create tensor on CPU first
        ramp = torch.linspace(0, 1, num_timesteps)
        min_inv_rho = sigma_min ** (1/rho)
        max_inv_rho = sigma_max ** (1/rho)
        
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas[0] = sigma_min  # First step
        sigmas[-1] = sigma_max  # ZTSNR step
        
        # Move to device at the end
        sigmas = sigmas.to(device=device)
        logger.info(f"Generated sigmas on device: {sigmas.device}")
        
        return sigmas
        
    except Exception as e:
        logger.error(f"Error in get_sigmas: {str(e)}")
        raise

def configure_noise_scheduler(config: "Config", device: torch.device) -> Dict[str, Any]:
    """Configure noise scheduler with Karras schedule and pre-compute training parameters."""
    try:
        if not isinstance(config, Config):
            raise ValueError(f"Expected config to be Config, got {type(config)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"Expected device to be torch.device, got {type(device)}")
            
        # Normalize device specification
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda:0')
            
        # Initialize scheduler
        try:
            scheduler = DDPMScheduler(
                num_train_timesteps=config.model.num_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=config.training.prediction_type,
                clip_sample=False,
                thresholding=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize DDPMScheduler: {str(e)}")
            raise
        
        # Generate sigmas and compute parameters
        try:
            sigmas = get_sigmas(config, device)
            params = get_scheduler_parameters(sigmas, config, device)
            
            # Move parameters to device directly without CPU intermediary
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    params[key] = value.to(device=device)
                    logger.info(f"Moved scheduler parameter {key} to device {device}")
            
            # Verify all parameters are on correct device
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    if str(value.device) != str(device):  # Compare device strings to handle cuda:0 vs cuda
                        logger.error(f"Parameter {key} is on {value.device} but should be on {device}")
                        raise RuntimeError(f"Failed to move scheduler parameter {key} to device {device}")
                    
        except Exception as e:
            logger.error(f"Failed to generate scheduler parameters: {str(e)}")
            raise
        
        # Update scheduler with computed values
        try:
            scheduler.alphas = params['alphas'].clone()
            scheduler.betas = params['betas'].clone()
            scheduler.alphas_cumprod = params['alphas_cumprod'].clone()
            scheduler.sigmas = sigmas.clone()
            scheduler.init_noise_sigma = float(sigmas.max().item())  # Convert to float to avoid device issues
            
            # Verify scheduler parameters
            if not isinstance(scheduler.alphas, torch.Tensor):
                raise RuntimeError(f"Scheduler alphas not a tensor")
            if not isinstance(scheduler.betas, torch.Tensor):
                raise RuntimeError(f"Scheduler betas not a tensor")
            if not isinstance(scheduler.alphas_cumprod, torch.Tensor):
                raise RuntimeError(f"Scheduler alphas_cumprod not a tensor")
            if not isinstance(scheduler.sigmas, torch.Tensor):
                raise RuntimeError(f"Scheduler sigmas not a tensor")
                
        except Exception as e:
            logger.error(f"Failed to update scheduler parameters: {str(e)}")
            raise
        
        # Return all parameters including scheduler
        return {'scheduler': scheduler, **params}

    except Exception as e:
        logger.error(f"Failed to configure noise scheduler: {str(e)}")
        raise
