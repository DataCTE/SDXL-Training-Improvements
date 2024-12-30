"""Base trainer implementation with common functionality."""
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import time
from abc import ABC, abstractmethod
import json

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.core.distributed import is_main_process
from src.data.utils.paths import convert_windows_path
from src.core.logging import WandbLogger
from safetensors.torch import save_file, load_file

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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_logger = wandb_logger
        self.config = config or {}
        
        # Store DataLoader configuration
        self._dataloader_config = {
            'dataset': train_dataloader.dataset,
            'batch_size': train_dataloader.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': train_dataloader.num_workers,
            'pin_memory': train_dataloader.pin_memory,
        }
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize metrics tracking
        self.train_metrics = {}
        self.eval_metrics = {}
        
        # Set up device
        self.model.to(self.device)
        
        # Additional setup from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    @property
    def train_dataloader(self):
        """Create new DataLoader instance when accessed."""
        return torch.utils.data.DataLoader(**self._dataloader_config)
        
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
        """Save training checkpoint using safetensors format."""
        try:
            # Convert .pt extension to .safetensors if present
            save_path = convert_windows_path(path)
            if save_path.endswith('.pt'):
                save_path = save_path[:-3] + '.safetensors'
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare tensors for safetensors format
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                # Convert scalar values to tensors
                "step": torch.tensor(self.step),
                "epoch": torch.tensor(self.epoch),
                "global_step": torch.tensor(self.global_step),
                "best_loss": torch.tensor(self.best_loss),
                "timestamp": torch.tensor(time.time())
            }
            
            # Add any additional tensor data
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    checkpoint[k] = v
                else:
                    try:
                        checkpoint[k] = torch.tensor(v)
                    except Exception:
                        logger.warning(f"Skipping non-tensor kwargs item: {k}")
            
            # Save using safetensors
            save_file(checkpoint, save_path)
            
            # Save config separately as it can't be stored in safetensors
            config_path = Path(save_path).with_suffix('.config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise
            
    def load_checkpoint(self, path: str):
        """Load training checkpoint from safetensors format."""
        try:
            # Convert .pt extension to .safetensors if present
            load_path = convert_windows_path(path)
            if load_path.endswith('.pt'):
                load_path = load_path[:-3] + '.safetensors'
            if not Path(load_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {load_path}")
                
            # Load tensors using safetensors
            checkpoint = load_file(load_path)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
            # Load scalar values
            self.step = checkpoint["step"].item()
            self.epoch = checkpoint["epoch"].item()
            self.global_step = checkpoint["global_step"].item()
            self.best_loss = checkpoint["best_loss"].item()
            
            # Load config from separate file if it exists
            config_path = Path(load_path).with_suffix('.config.json')
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
            
            logger.info(f"Loaded checkpoint from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
            raise 