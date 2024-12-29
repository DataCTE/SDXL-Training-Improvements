"""Base classes for SDXL training methods with extreme speedups."""
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type, Tuple
import torch
import torch.backends.cudnn
from src.data.config import Config
from src.training.schedulers import configure_noise_scheduler
from src.core.logging import get_logger

class TrainingMethodMeta(ABCMeta):
    _methods: Dict[str, Type['TrainingMethod']] = {}
    _logger = get_logger("training.methods")

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'name' in attrs:
            mcs._methods[attrs['name']] = cls
            mcs._logger.debug(f"Registered training method: {attrs['name']}")
        return cls
    
    @classmethod
    def get_method(mcs, name: str) -> Type['TrainingMethod']:
        if name not in mcs._methods:
            mcs._logger.error(
                f"Unknown training method: {name}",
                extra={'available_methods': list(mcs._methods.keys())}
            )
            raise ValueError(
                f"Unknown training method: {name}. "
                f"Available methods: {list(mcs._methods.keys())}"
            )
        mcs._logger.debug(f"Retrieved training method: {name}")
        return mcs._methods[name]

class TrainingMethod(metaclass=TrainingMethodMeta):
    name: str = None
    
    def __init__(self, unet: torch.nn.Module, config: Config):
        # Get logger with proper configuration
        self.logger = get_logger(f"training.methods.{self.name}")
        self.tensor_logger = get_logger(f"training.methods.{self.name}.tensor")
        
        self.logger.debug(f"Initializing {self.__class__.__name__}")
        # Add memory optimization logging
        self.logger.debug(f"CUDA settings: benchmark={torch.backends.cudnn.benchmark}, "
                         f"allow_tf32={torch.backends.cudnn.allow_tf32}, "
                         f"memory_allocated={torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            self.logger.debug("CUDA optimizations enabled")

        self.unet = unet
        self.config = config
        self.training = True
        
        # Initialize noise scheduler using the configuration
        try:
            self.noise_scheduler = configure_noise_scheduler(config, self.unet.device)
            self.logger.debug("Noise scheduler configured successfully")
        except Exception as e:
            self.logger.error(
                "Failed to configure noise scheduler",
                exc_info=True,
                extra={
                    'error': str(e),
                    'device': str(self.unet.device),
                    'config': str(config)
                }
            )
            raise
