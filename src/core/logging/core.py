"""Core logging functionality with integrated features."""
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import torch
import wandb
from tqdm import tqdm
import numpy as np
from functools import wraps
from dataclasses import dataclass, field
from collections import deque

from .logger import ColoredFormatter, LogConfig, ProgressTracker, MetricsTracker

class UnifiedLogger:
    """Single entry point for all logging functionality."""
    
    def __init__(self, config: Optional[Union[LogConfig, Dict]] = None):
        """Initialize unified logger with optional configuration."""
        self.config = LogConfig(**config) if isinstance(config, dict) else config or LogConfig()
        self._setup_base_logger()
        self._setup_tracking_features()
        
    def _setup_base_logger(self) -> None:
        """Initialize base logger with configured handlers."""
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Console handler setup
        if self.config.enable_console:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(getattr(logging, self.config.console_level.upper()))
            console.setFormatter(ColoredFormatter())
            self.logger.addHandler(console)
            
        # File handler setup
        if self.config.enable_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file = logging.FileHandler(
                log_dir / self.config.filename,
                encoding='utf-8'
            )
            file.setLevel(getattr(logging, self.config.file_level.upper()))
            file.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            ))
            self.logger.addHandler(file)
            
    def _setup_tracking_features(self) -> None:
        """Initialize tracking features based on configuration."""
        # W&B initialization
        self._wandb_run = None
        if self.config.enable_wandb:
            try:
                self._wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_name,
                    tags=self.config.wandb_tags,
                    config=vars(self.config)
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                
        # Progress tracking setup
        self._progress = None
        if self.config.enable_progress:
            self._progress = ProgressTracker(
                window_size=self.config.progress_window,
                smoothing=self.config.progress_smoothing
            )
            
        # Metrics tracking setup
        self._metrics = MetricsTracker(
            window_size=self.config.metrics_window,
            keep_history=self.config.metrics_history
        )
    
    # Core logging methods
    def debug(self, msg: str, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def info(self, msg: str, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg: str, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg: str, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics with optional W&B integration."""
        if self.metrics:
            self.metrics.update(metrics, step)
            
        if self.wandb:
            self.wandb.log(metrics, step=step)
            
    def update_progress(self, n: int = 1, **kwargs):
        """Update progress tracking."""
        if self.progress:
            self.progress.update(n, **kwargs)
            
    def finish(self):
        """Cleanup and close all logging systems."""
        if self.wandb:
            self.wandb.finish()
            
        if self.progress:
            self.progress.close()
            
        if self.metrics:
            self.metrics.close() 
