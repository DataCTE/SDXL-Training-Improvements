"""Base classes for SDXL training methods with extreme speedups."""
import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type, Tuple
import torch
import torch.backends.cudnn
from torch import Tensor
from diffusers import DDPMScheduler
from src.data.config import Config
from src.training.schedulers import configure_noise_scheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure base class logger is at DEBUG level

class TrainingMethodMeta(ABCMeta):
    _methods: Dict[str, Type['TrainingMethod']] = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'name' in attrs:
            mcs._methods[attrs['name']] = cls
        return cls
    
    @classmethod
    def get_method(mcs, name: str) -> Type['TrainingMethod']:
        if name not in mcs._methods:
            raise ValueError(
                f"Unknown training method: {name}. "
                f"Available methods: {list(mcs._methods.keys())}"
            )
        return mcs._methods[name]

class TrainingMethod(metaclass=TrainingMethodMeta):
    name: str = None
    
    def __init__(self, unet: torch.nn.Module, config: Config):
        # Add logger verification at initialization
        logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        self.unet = unet
        self.config = config
        self.training = True
        
        # Initialize noise scheduler using the configuration
        self.noise_scheduler = configure_noise_scheduler(config, self.unet.device)
