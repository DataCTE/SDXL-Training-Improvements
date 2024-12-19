"""Base training method interface."""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

class TrainingMethod(ABC):
    """Abstract base class for training methods."""
    
    @abstractmethod
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        noise_scheduler: Optional[object] = None,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for a batch.
        
        Args:
            model: The model being trained
            batch: Batch of training data
            noise_scheduler: Optional noise scheduler
            generator: Optional random number generator
            
        Returns:
            Dict containing loss tensor and any auxiliary metrics
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get method name."""
        pass
