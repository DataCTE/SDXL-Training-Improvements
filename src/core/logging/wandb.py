"""Weights & Biases logging utilities."""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import functools
from functools import wraps

import torch
import wandb
from PIL import Image
import logging
from src.core.logging.base import LogConfig
from src.data.config import Config
from src.training.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


def make_picklable(func):
    """Decorator to make functions picklable."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class WandbLogger:
    """Weights & Biases logger for SDXL training."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: bool = False,
        log_config: Optional[LogConfig] = None,
    ):
        """Initialize W&B logger.
        
        Args:
            config: Main configuration object
            project: Override project name (default: from config)
            name: Override run name (default: from config)
            tags: Override run tags (default: from config)
            notes: Override run notes
            resume: Whether to resume previous run
            log_config: Legacy logging configuration
        """
        self.enabled = True
        self.model_logged = False
        
        try:
            if not self.enabled:
                logger.warning("W&B logging disabled")
                return

            # Priority: log_config > explicit params > config > defaults
            wandb_config = {}
            
            # 1. Start with defaults from Config class
            default_config = Config()
            wandb_project = default_config.global_config.logging.wandb_project
            wandb_entity = default_config.global_config.logging.wandb_entity
            
            # 2. Update from provided config if available
            if config:
                wandb_project = config.global_config.logging.wandb_project
                wandb_entity = config.global_config.logging.wandb_entity
                wandb_config = config.to_dict() if hasattr(config, 'to_dict') else {}
            
            # 3. Update from explicit parameters
            if project:
                wandb_project = project
            
            # 4. Update from log_config if provided (legacy support)
            if log_config:
                if hasattr(log_config, 'wandb_project'):
                    wandb_project = log_config.wandb_project
                if hasattr(log_config, 'wandb_name'):
                    name = log_config.wandb_name
                if hasattr(log_config, 'wandb_tags'):
                    tags = log_config.wandb_tags
                if hasattr(log_config, 'wandb_notes'):
                    notes = log_config.wandb_notes
                
            self.run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=name,
                config=wandb_config,
                tags=tags,
                notes=notes,
                resume=resume
            )

            # Print the run URL immediately after initialization
            if self.run is not None:
                print(f"\n🔗 Weights & Biases run: {self.run.get_url()}\n")
                logger.info(f"Initialized W&B run: {self.run.get_url()}")
            
        except Exception as e:
            logger.error(
                "Failed to initialize W&B",
                exc_info=True,
                extra={
                    'error': str(e),
                    'project': wandb_project,
                    'enabled': self.enabled
                }
            )
            self.enabled = False

    @make_picklable
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log metrics to W&B."""
        if not self.enabled:
            return
            
        try:
            # Validate metrics
            filtered_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float, str, bool)):
                        filtered_metrics[k] = v
                    elif isinstance(v, torch.Tensor):
                        filtered_metrics[k] = v.item()
                except Exception as e:
                    logger.warning(f"Failed to process metric {k}: {str(e)}")
                    
            wandb.log(filtered_metrics, step=step, commit=commit)
            
        except Exception as e:
            logger.error(
                "Failed to log metrics",
                exc_info=True,
                extra={
                    'error': str(e),
                    'step': step,
                    'metrics_keys': list(metrics.keys())
                }
            )

    @make_picklable
    def log_images(
        self,
        images: Dict[str, Union[Image.Image, torch.Tensor]],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log images to W&B."""
        if not self.enabled:
            return
            
        try:
            processed_images = {}
            for name, img in images.items():
                if isinstance(img, torch.Tensor):
                    if img.ndim == 4:
                        img = img[0]
                    img = (img * 255).byte().cpu().numpy()
                    if img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)
                    img = Image.fromarray(img)
                processed_images[name] = wandb.Image(img)
                
            wandb.log(processed_images, step=step, commit=commit)
        except Exception as e:
            logger.error(f"Failed to log images: {str(e)}")

    def log_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional["BaseOptimizer"] = None,
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log model architecture and gradients to W&B."""
        if not self.enabled:
            return
            
        try:
            # Import here to avoid circular imports
            from src.training.optimizers.base import BaseOptimizer
            
            if not hasattr(self, 'model_logged') or not self.model_logged:
                # Log model info
                model_config = {
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'model_structure': str(model),
                }
                
                # Add optimizer info if provided and valid type
                if optimizer and isinstance(optimizer, BaseOptimizer):
                    model_config['optimizer'] = optimizer.__class__.__name__
                    model_config['optimizer_state'] = str(optimizer.state_dict().keys())
                
                wandb.config.update({'model': model_config}, allow_val_change=True)
                
                # Log parameter stats
                param_stats = {
                    f"parameters/{name}_norm": param.norm().item()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                wandb.log(param_stats, step=step, commit=commit)
                
                self.model_logged = True
                
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")

    @make_picklable
    def log_hyperparams(
        self,
        params: Dict[str, Any],
        commit: bool = True
    ) -> None:
        """Log hyperparameters to W&B."""
        if not self.enabled:
            return
            
        try:
            wandb.config.update(params, allow_val_change=True)
            if commit:
                wandb.log({})
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {str(e)}")

    @make_picklable
    def finish(self) -> None:
        """Finish logging and close W&B run."""
        if not self.enabled:
            return
            
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {str(e)}")
