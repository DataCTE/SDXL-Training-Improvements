"""Base trainer implementation with common functionality."""
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import time
from abc import ABC, abstractmethod

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.core.distributed import is_main_process
from src.data.utils.paths import convert_windows_path
from src.core.logging import WandbLogger

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """Abstract base trainer with common functionality."""
    
    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        wandb_logger: Optional[WandbLogger] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """Initialize base trainer.
        
        Args:
            model: SDXL model instance
            optimizer: Optimizer instance
            train_dataloader: Training data loader
            device: Torch device to use
            wandb_logger: Optional W&B logger
            config: Training configuration
            **kwargs: Additional arguments
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_logger = wandb_logger
        self.config = config or {}
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Get noise scheduler from model
        self.noise_scheduler = model.noise_scheduler
        
        # Initialize metrics tracking
        self.train_metrics = {}
        self.eval_metrics = {}
        
        # Set up device
        self.model.to(self.device)
        
        # Additional setup from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    @abstractmethod
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Execute single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dict containing loss and metrics
        """
        raise NotImplementedError
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        for batch in self.train_dataloader:
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.training_step(batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                metrics = outputs.get("metrics", {})
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v)
                    
                self.step += 1
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}", exc_info=True)
                continue
                
        # Compute epoch metrics
        epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        
        # Log metrics
        if self.wandb_logger and is_main_process():
            self.wandb_logger.log_metrics({
                f"epoch_{k}": v for k, v in epoch_metrics.items()
            })
            
        return epoch_metrics
        
    def train(self, num_epochs: int):
        """Train for specified number of epochs."""
        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                
                # Train epoch
                metrics = self.train_epoch()
                
                # Log epoch results
                if is_main_process():
                    logger.info(f"Epoch {epoch} metrics:")
                    for k, v in metrics.items():
                        logger.info(f"{k}: {v:.4f}")
                        
                # Save checkpoint if improved
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]
                    self.save_checkpoint("best_model.pt")
                    
                # Regular checkpoint
                if (epoch + 1) % self.config.get("save_every", 1) == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                    
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise
            
    def save_checkpoint(self, path: str, **kwargs):
        """Save training checkpoint."""
        try:
            save_path = convert_windows_path(path)
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "config": self.config,
                "timestamp": time.time()
            }
            
            # Add any additional data
            checkpoint.update(kwargs)
            
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise
            
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        try:
            load_path = convert_windows_path(path)
            if not Path(load_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {load_path}")
                
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.global_step = checkpoint.get("global_step", self.step)
            self.best_loss = checkpoint.get("best_loss", float('inf'))
            
            logger.info(f"Loaded checkpoint from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
            raise 