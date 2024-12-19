"""Base training method interface for SDXL."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor
from diffusers import DDPMScheduler

from src.data.config import Config

class TrainingMethod(ABC):
    """Abstract base class for SDXL training methods."""
    
    def __init__(self, unet: torch.nn.Module, config: Config):
        """Initialize training method.
        
        Args:
            unet: UNet model
            config: Training configuration
        """
        self.unet = unet
        self.config = config
        self.training = True
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.model.num_timesteps,
            prediction_type=config.training.prediction_type,
            rescale_betas_zero_snr=config.training.zero_terminal_snr
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Get training method name."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss.
        
        Args:
            model: UNet model
            batch: Training batch
            generator: Optional random generator
            
        Returns:
            Dict containing loss and any additional metrics
        """
        pass
        
    def train(self) -> None:
        """Set training mode."""
        self.training = True
        
    def eval(self) -> None:
        """Set evaluation mode."""
        self.training = False
