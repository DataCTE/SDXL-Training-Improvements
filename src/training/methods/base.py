"""Base classes for SDXL training methods with extreme speedups."""
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type
from src.core.history import TorchHistory
import torch
import torch.backends.cudnn
import functools
from torch import Tensor
from diffusers import DDPMScheduler
from src.data.config import Config



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

def make_picklable(func):
    """Decorator to make functions picklable."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class TrainingMethod(metaclass=TrainingMethodMeta):
    name: str = None

    def __init__(self, unet: torch.nn.Module, config: Config):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        self.unet = unet
        self.config = config
        self.training = True
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.model.num_timesteps,
            prediction_type=config.training.prediction_type,
            rescale_betas_zero_snr=config.training.zero_terminal_snr
        )
        # Move scheduler tensors to device
        for attr_name in dir(self.noise_scheduler):
            if attr_name.startswith('__'):
                continue
            attr = getattr(self.noise_scheduler, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self.noise_scheduler, attr_name, attr.to(unet.device))

    @abstractmethod
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        pass
        
    def train(self) -> None:
        self.training = True
        
    def eval(self) -> None:
        self.training = False
