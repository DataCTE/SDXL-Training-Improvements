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
        
        # Check if dataset can be pickled
        try:
            import pickle
            pickle.dumps(train_dataloader.dataset)
            can_pickle = True
        except:
            can_pickle = False
            logger.warning("Dataset cannot be pickled, falling back to single-process data loading")
        
        # Define custom collate function that handles different sized tensors
        def dynamic_collate(batch):
            """Custom collate function that handles tensors of different sizes."""
            error_context = {}
            try:
                # Filter out None values
                batch = [b for b in batch if b is not None]
                if not batch:
                    raise RuntimeError("Empty batch after filtering")
                
                # Separate different keys
                elem = batch[0]
                if not isinstance(elem, dict):
                    raise TypeError(f"Expected dict but got {type(elem)}")
                
                result = {}
                for key in elem:
                    error_context['current_key'] = key
                    if isinstance(elem[key], torch.Tensor):
                        # Move tensors to CPU if they're on CUDA
                        tensors = [b[key].cpu() if b[key].is_cuda else b[key] for b in batch]
                        
                        # Get target size from config
                        if (hasattr(self.config, 'global_config') and 
                            hasattr(self.config.global_config, 'image') and
                            hasattr(self.config.global_config.image, 'target_size')):
                            target_size = self.config.global_config.image.target_size
                        else:
                            # Use minimum dimensions as fallback
                            sizes = [t.size() for t in tensors]
                            target_size = [min(s[i] for s in sizes) for i in range(2)]
                        
                        # Resize tensors to target size if they don't match
                        resized_tensors = []
                        for t in tensors:
                            if len(t.shape) >= 3:  # Only resize if tensor has spatial dimensions
                                if t.size(-2) != target_size[0] or t.size(-1) != target_size[1]:
                                    # Preserve batch dimension and channels
                                    t = torch.nn.functional.interpolate(
                                        t.unsqueeze(0) if len(t.shape) == 3 else t,
                                        size=target_size,
                                        mode='bilinear',
                                        align_corners=False
                                    )
                                    if len(t.shape) == 4:  # Remove batch dimension if it was added
                                        t = t.squeeze(0)
                            resized_tensors.append(t)
                        
                        # Stack the resized tensors
                        try:
                            result[key] = torch.stack(resized_tensors)
                        except Exception as e:
                            error_context.update({
                                'tensor_shapes': [t.size() for t in resized_tensors],
                                'target_size': target_size,
                                'key': key
                            })
                            raise RuntimeError(f"Failed to stack tensors for key {key}: {str(e)}") from e
                    else:
                        # For non-tensors, use simple list
                        result[key] = [b[key] for b in batch]
                
                return result
                
            except Exception as e:
                error_context['batch_sizes'] = [
                    {k: v.size() if isinstance(v, torch.Tensor) else len(v) for k, v in b.items()}
                    for b in batch
                ] if batch else []
                logger.error(f"Collate failed: {str(e)}", extra=error_context)
                raise RuntimeError(f"Collate failed: {str(e)}") from e
        
        # Store DataLoader configuration with custom collate
        self._dataloader_config = {
            'dataset': train_dataloader.dataset,
            'batch_size': train_dataloader.batch_size,
            'collate_fn': dynamic_collate,
            'shuffle': True,
            'drop_last': getattr(train_dataloader, 'drop_last', True),
            'num_workers': train_dataloader.num_workers if can_pickle else 0,
            # Only enable pin_memory if using CPU tensors and CUDA is available
            'pin_memory': (train_dataloader.pin_memory and 
                         torch.cuda.is_available() and 
                         not any(isinstance(getattr(train_dataloader.dataset, attr, None), torch.Tensor) and 
                                getattr(train_dataloader.dataset, attr, None).is_cuda
                                for attr in dir(train_dataloader.dataset))),
        }
        
        # Log DataLoader configuration
        logger.info(f"DataLoader config: workers={self._dataloader_config['num_workers']}, "
                   f"pin_memory={self._dataloader_config['pin_memory']}")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_logger = wandb_logger
        self.config = config or {}
        
        # Get noise scheduler from model and initialize it
        self.noise_scheduler = model.noise_scheduler
        
        # Configure scheduler based on config
        if (hasattr(self.config, 'training') and 
            hasattr(self.config.training, 'method_config') and 
            hasattr(self.config.training.method_config, 'scheduler')):
            scheduler_config = self.config.training.method_config.scheduler
            # Convert scheduler config to dictionary if it's not already
            if not isinstance(scheduler_config, dict):
                scheduler_config = scheduler_config.__dict__ if hasattr(scheduler_config, '__dict__') else {}
            
            for key, value in scheduler_config.items():
                if hasattr(self.noise_scheduler, key):
                    try:
                        setattr(self.noise_scheduler, key, value)
                        logger.info(f"Set scheduler {key}={value}")
                    except Exception as e:
                        logger.warning(f"Failed to set scheduler {key}={value}: {str(e)}")
        
        # Set number of inference steps
        num_inference_steps = (
            self.config.training.num_inference_steps 
            if hasattr(self.config, 'training') and hasattr(self.config.training, 'num_inference_steps')
            else 1000
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
        logger.info(f"Set noise scheduler timesteps to {num_inference_steps}")
            
        # Store timesteps after initialization
        self.timesteps = self.noise_scheduler.timesteps.to(self.device)
        
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
                save_every = (self.config.training.save_every 
                            if hasattr(self.config, 'training') and hasattr(self.config.training, 'save_every')
                            else 1)
                if (epoch + 1) % save_every == 0:
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

    def get_timestep(self, batch_size: int) -> torch.Tensor:
        """Get timesteps for the current training step."""
        if not hasattr(self, 'timesteps'):
            raise RuntimeError("Timesteps not initialized. Did you forget to set_timesteps?")
            
        # Sample timesteps uniformly
        indices = torch.randint(0, len(self.timesteps), (batch_size,), device=self.device)
        return self.timesteps[indices] 