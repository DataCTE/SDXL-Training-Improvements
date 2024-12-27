"""Weights & Biases logging utilities."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import functools
from functools import wraps

import functools
import torch
import wandb
from PIL import Image

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
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[Union[str, Path]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: bool = False
    ):
        """Initialize W&B logger."""
        self.enabled = True
        self.model_logged = False
        try:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=dir,
                tags=tags,
                notes=notes,
                resume=resume
            )
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {str(e)}")
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
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")

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
        criterion: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """Log model architecture and gradients to W&B."""
        if not self.enabled:
            return
            
        try:
            # Only watch model once
            if not hasattr(self, 'model_logged') or not self.model_logged:
                wandb.watch(
                    model,
                    criterion=criterion,
                    log="all",
                    log_freq=100
                )
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
