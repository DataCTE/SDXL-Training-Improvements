"""Base training method interface for SDXL."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor

class TrainingMethod(ABC):
    """Abstract base class for SDXL training methods."""
    
    def __init__(self, unet: torch.nn.Module, config: "Config"):  # type: ignore
        """Initialize training method.
        
        Args:
            unet: UNet model
            config: Training configuration
        """
        self.unet = unet
        self.config = config
        self.training = True

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
