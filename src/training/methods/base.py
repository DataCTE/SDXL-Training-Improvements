"""Base classes for SDXL training methods."""
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type

import torch
from torch import Tensor
from diffusers import DDPMScheduler

from src.data.config import Config

class TrainingMethodMeta(ABCMeta):
    """Metaclass for training methods to handle registration."""
    
    _methods: Dict[str, Type['TrainingMethod']] = {}
    
    def __new__(mcs, name, bases, attrs):
        """Create new training method class and register it."""
        cls = super().__new__(mcs, name, bases, attrs)
        if 'name' in attrs:  # Only register if name is explicitly defined
            mcs._methods[attrs['name']] = cls
        return cls
    
    @classmethod
    def get_method(mcs, name: str) -> Type['TrainingMethod']:
        """Get training method class by name."""
        if name not in mcs._methods:
            raise ValueError(
                f"Unknown training method: {name}. "
                f"Available methods: {list(mcs._methods.keys())}"
            )
        return mcs._methods[name]

class TrainingMethod(metaclass=TrainingMethodMeta):
    """Abstract base class for SDXL training methods."""
    
    name: str = None  # Must be defined by subclasses
    
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
        self.noise_scheduler.to(unet.device)

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
