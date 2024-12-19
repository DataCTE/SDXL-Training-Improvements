"""Noise scheduler and related functions for SDXL training."""
import logging
import math
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from ..data.config import Config

logger = logging.getLogger(__name__)

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        timestep_spacing: str = "leading",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        clip_sample_range: float = 1.0,
        num_inference_steps: int = 50,
    ):
        """Initialize DDPM scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Final beta value
            beta_schedule: Type of beta schedule
            clip_sample: Whether to clip samples
            prediction_type: Type of prediction (epsilon/sample/v)
            set_alpha_to_one: Whether to force alpha=1
            steps_offset: Offset for step indices
            timestep_spacing: How to space timesteps
            thresholding: Whether to use dynamic thresholding
            dynamic_thresholding_ratio: Ratio for dynamic thresholding
            sample_max_value: Maximum sample value
            clip_sample_range: Sample clipping range
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.timestep_spacing = timestep_spacing
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.clip_sample_range = clip_sample_range

        # Initialize betas and alphas
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Set final alpha if needed
        if self.set_alpha_to_one:
            self.alphas_cumprod[-1] = 1.0
            
        # Compute derived values
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.init_noise_sigma = self.sigmas.max()

    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule tensor."""
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            # Glide paper schedule
            betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        elif self.beta_schedule == "squaredcos_cap_v2":
            # Stable Diffusion schedule
            t = torch.linspace(0, self.num_train_timesteps - 1, self.num_train_timesteps)
            betas = 0.999 * (torch.cos((t / self.num_train_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        return betas

    def scale_model_input(self, sample: torch.Tensor, timestep: int) -> torch.Tensor:
        """Scale input sample based on timestep.
        
        Args:
            sample: Input tensor to scale
            timestep: Current timestep
            
        Returns:
            Scaled input tensor
        """
        # Get sigma for timestep
        step_index = (self.sigmas == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        
        # Scale input
        scaled = sample / ((sigma ** 2 + 1) ** 0.5)
        return scaled

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        """Set timesteps for inference.
        
        Args:
            num_inference_steps: Optional number of inference steps to override default
        """
        self.num_inference_steps = num_inference_steps or self.num_inference_steps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Predict previous mean and variance."""
        if not hasattr(self, 'num_inference_steps'):
            raise ValueError("Number of inference steps not set. Call set_timesteps() first.")
            
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else 1.0
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        # Clip predicted sample if needed
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        # Get previous sample
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + beta_prod_t_prev ** 0.5 * model_output

        if not return_dict:
            return (prev_sample,)

        return {"prev_sample": prev_sample}

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples."""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get velocity for v-prediction."""
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

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

def get_scheduler_parameters(
    sigmas: torch.Tensor,
    config: "Config",
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Generate scheduler parameters from sigma schedule.
    
    Args:
        sigmas: Noise schedule sigmas
        config: Training config
        device: Target device
        
    Returns:
        Dict of scheduler parameters
    """
    try:
        # Convert sigmas to betas using config parameters
        sigmas = sigmas.to(device)
        sigmas_next = torch.cat([sigmas[1:], torch.tensor([config.model.sigma_min], device=device)])
        
        # Compute alphas and betas with config-based clipping
        alphas = 1 / (1 + sigmas ** 2)
        betas = 1 - alphas / (1 + sigmas_next ** 2)
        betas = torch.clip(betas, 0, config.training.snr_gamma or 0.999)
        
        # Compute alphas cumprod
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            'alphas': alphas,
            'betas': betas,
            'alphas_cumprod': alphas_cumprod,
            'sigmas': sigmas
        }
        
    except Exception as e:
        logger.error(f"Error generating scheduler parameters: {str(e)}")
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
