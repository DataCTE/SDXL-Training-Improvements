"""Base classes for SDXL training methods with extreme speedups."""
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type, Tuple
import torch
import torch.backends.cudnn
from torch import Tensor
from diffusers import DDPMScheduler
from src.data.config import Config
from src.training.schedulers import configure_noise_scheduler
from src.core.logging import setup_logging, LoggingConfig

class TrainingMethodMeta(ABCMeta):
    _methods: Dict[str, Type['TrainingMethod']] = {}
    _logger = None

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if 'name' in attrs:
            mcs._methods[attrs['name']] = cls
            # Setup method-specific logger
            if not mcs._logger:
                mcs._logger, _ = setup_logging(
                    module_name="training.methods",
                    console_level="DEBUG"
                )
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
        self.logger, self.tensor_logger = setup_logging(
            module_name=f"training.methods.{self.name}",
            config=config.logging.to_core_config()
        )
        
        self.logger.debug(f"Initializing {self.__class__.__name__}")
        self.logger.debug(f"CUDA settings: benchmark={torch.backends.cudnn.benchmark}, "
                         f"allow_tf32={torch.backends.cudnn.allow_tf32}")
        
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
