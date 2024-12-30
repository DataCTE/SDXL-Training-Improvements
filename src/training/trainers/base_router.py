from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.core.logging import WandbLogger
from src.data.config import Config

class BaseRouter(ABC):
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: DataLoader,
        device: torch.device,
        wandb_logger: Optional[WandbLogger] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.wandb_logger = wandb_logger
        self.config = config

    @abstractmethod
    def train(self, num_epochs: int):
        """Abstract method to be implemented by specific architecture routers."""
        pass 