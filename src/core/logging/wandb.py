"""Weights & Biases logging utilities."""
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import wandb
import torch
from PIL import Image
import numpy as np

from .base import LogConfig
from .core import UnifiedLogger

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    # Basic settings
    project: str = "sdxl-training"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    
    # Run settings
    dir: Optional[str] = None
    resume: Union[bool, str] = False
    reinit: bool = False
    mode: str = "online"
    
    # Sync settings
    sync_tensorboard: bool = False
    save_code: bool = True
    save_requirements: bool = True
    
    # Media settings
    log_model: bool = True
    log_checkpoints: bool = True
    checkpoint_prefix: str = "checkpoint"
    sample_prefix: str = "sample"
    max_images_to_log: int = 16
    
    # Metric settings
    metric_prefix: str = "metrics/"
    log_system_metrics: bool = True
    system_metrics_interval: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for W&B init."""
        return {
            "project": self.project,
            "name": self.name,
            "tags": self.tags,
            "notes": self.notes,
            "group": self.group,
            "job_type": self.job_type,
            "dir": self.dir,
            "resume": self.resume,
            "reinit": self.reinit,
            "mode": self.mode,
            "sync_tensorboard": self.sync_tensorboard,
            "save_code": self.save_code
        }

class WandbLogger:
    """Enhanced Weights & Biases logger with SDXL-specific features."""
    
    def __init__(
        self,
        config: Optional[Union[WandbConfig, Dict]] = None,
        logger: Optional[UnifiedLogger] = None
    ):
        """Initialize W&B logger.
        
        Args:
            config: W&B configuration
            logger: Optional UnifiedLogger instance for additional logging
        """
        self.config = (
            WandbConfig(**config) if isinstance(config, dict)
            else config or WandbConfig()
        )
        self.logger = logger
        self._run = None
        self._last_system_metrics = 0
        
    def setup(self) -> None:
        """Initialize W&B run."""
        if self._run is not None:
            return
            
        # Initialize run
        self._run = wandb.init(**self.config.to_dict())
        
        if self.logger:
            self.logger.info(
                f"Initialized W&B run: {self._run.name} "
                f"(project: {self.config.project}, "
                f"mode: {self.config.mode})"
            )
            
        # Save pip requirements if enabled
        if self.config.save_requirements:
            self._save_requirements()
            
    def _save_requirements(self) -> None:
        """Save pip requirements to W&B."""
        try:
            import pkg_resources
            requirements = [
                f"{dist.key}=={dist.version}"
                for dist in pkg_resources.working_set
            ]
            self._run.save("requirements.txt", "\n".join(requirements))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save requirements: {e}")
                
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
        prefix: Optional[str] = None
    ) -> None:
        """Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional global step for metrics
            commit: Whether to immediately sync with W&B servers
            prefix: Optional prefix to override config.metric_prefix
        """
        if self._run is None:
            self.setup()
            
        # Convert values to float/int if possible
        processed_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = v.item() if hasattr(v, 'item') else float(v)
            processed_metrics[k] = v
            
        # Add metric prefix if configured
        metric_prefix = prefix or self.config.metric_prefix
        if metric_prefix:
            processed_metrics = {
                f"{metric_prefix}{k}": v
                for k, v in processed_metrics.items()
            }
            
        # Log system metrics if enabled
        if (
            self.config.log_system_metrics and
            time.time() - self._last_system_metrics > self.config.system_metrics_interval
        ):
            processed_metrics.update(self._get_system_metrics())
            self._last_system_metrics = time.time()
            
        # Log to W&B
        self._run.log(processed_metrics, step=step, commit=commit)
        
        # Log to UnifiedLogger if available
        if self.logger:
            self.logger.log_metrics(metrics, step=step)
                
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (GPU, CPU, memory)."""
        metrics = {}
        
        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics.update({
                    f"system/gpu{i}/memory_allocated": torch.cuda.memory_allocated(i) / 1e9,
                    f"system/gpu{i}/memory_reserved": torch.cuda.memory_reserved(i) / 1e9,
                    f"system/gpu{i}/utilization": torch.cuda.utilization(i)
                })
                
        # CPU metrics
        try:
            import psutil
            metrics.update({
                "system/cpu/percent": psutil.cpu_percent(),
                "system/memory/percent": psutil.virtual_memory().percent,
                "system/disk/percent": psutil.disk_usage("/").percent
            })
        except ImportError:
            pass
            
        return metrics
        
    def log_images(
        self,
        images: Union[List[Image.Image], List[np.ndarray], List[torch.Tensor]],
        captions: Optional[List[str]] = None,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        commit: bool = True
    ) -> None:
        """Log images to W&B.
        
        Args:
            images: List of images (PIL, numpy array, or torch tensor)
            captions: Optional list of captions for each image
            step: Optional global step
            prefix: Optional prefix for image names
            commit: Whether to immediately sync with W&B servers
        """
        if self._run is None:
            self.setup()
            
        # Limit number of images
        if len(images) > self.config.max_images_to_log:
            if self.logger:
                self.logger.warning(
                    f"Limiting number of logged images from {len(images)} "
                    f"to {self.config.max_images_to_log}"
                )
            images = images[:self.config.max_images_to_log]
            if captions:
                captions = captions[:self.config.max_images_to_log]
                
        # Convert images to W&B format
        wandb_images = []
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if isinstance(img, np.ndarray):
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            caption = captions[i] if captions else None
            wandb_images.append(wandb.Image(img, caption=caption))
            
        # Log images
        prefix = prefix or self.config.sample_prefix
        self._run.log(
            {f"{prefix}/images": wandb_images},
            step=step,
            commit=commit
        )
        
    def log_model(
        self,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Log model checkpoint to W&B.
        
        Args:
            model_path: Path to model checkpoint
            metadata: Optional metadata to log with model
            aliases: Optional list of aliases for the model
        """
        if not self.config.log_model:
            return
            
        if self._run is None:
            self.setup()
            
        # Convert path to string
        model_path = str(model_path)
        
        # Create artifact
        name = os.path.basename(model_path)
        artifact = wandb.Artifact(
            name=f"{self.config.checkpoint_prefix}-{name}",
            type="model",
            metadata=metadata
        )
        
        # Add file to artifact
        artifact.add_file(model_path)
        
        # Log artifact
        self._run.log_artifact(artifact, aliases=aliases)
        
        if self.logger:
            self.logger.info(f"Logged model checkpoint: {model_path}")
            
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration dictionary to W&B.
        
        Args:
            config: Configuration dictionary to log
        """
        if self._run is None:
            self.setup()
            
        # Update W&B config
        self._run.config.update(config)
        
    def finish(
        self,
        exit_code: int = 0,
        job_state: Optional[str] = None
    ) -> None:
        """Finish W&B run.
        
        Args:
            exit_code: Exit code to log
            job_state: Optional final job state to log
        """
        if self._run is not None:
            if job_state:
                self._run.log({"job_state": job_state})
            self._run.finish(exit_code=exit_code)
            self._run = None
            
    def __enter__(self) -> 'WandbLogger':
        """Context manager entry."""
        self.setup()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        exit_code = 1 if exc_type is not None else 0
        self.finish(exit_code=exit_code)