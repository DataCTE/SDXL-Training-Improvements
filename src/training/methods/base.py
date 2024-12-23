"""Base classes for SDXL training methods with extreme speedups."""
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Optional, Type
import torch
import torch.backends.cudnn
from torch import Tensor
from diffusers import DDPMScheduler
from src.data.config import Config

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

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
        self.unet = unet
        self.config = config
        self.training = True
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
