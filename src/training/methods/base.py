"""Base training method implementation."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor

class TrainingMethod(ABC):
    """Base class for all training methods."""
    
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
            model: Model to train
            batch: Training batch
            generator: Optional random number generator
            
        Returns:
            Dict containing loss and any additional metrics
        """
        pass
