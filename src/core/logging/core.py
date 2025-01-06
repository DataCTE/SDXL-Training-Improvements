"""Unified logging system with integrated features."""
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



class UnifiedLogger:
    """Single entry point for all logging functionality."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = self._setup_logger()
        self._setup_features()
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize base logger with handlers."""
        logger = logging.getLogger(self.config.name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            logger.addHandler(console_handler)
            
        if self.config.enable_file:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_path / self.config.filename,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.config.file_level.upper()))
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            ))
            logger.addHandler(file_handler)
            
        return logger
        
    def _setup_features(self):
        """Initialize optional features based on config."""
        # Initialize W&B
        self.wandb = None
        if self.config.enable_wandb:
            try:
                self.wandb = wandb.init(**self.config.wandb_config)
            except Exception as e:
                self.logger.error(f"Failed to initialize W&B: {e}")
                
        # Initialize progress tracking
        self.progress = None
        if self.config.enable_progress:
            self.progress = ProgressTracker(self.config.progress_config)
            
        # Initialize metrics tracking
        self.metrics = None
        if self.config.enable_metrics:
            self.metrics = MetricsTracker(
                self.config.metrics_config,
                self.wandb if self.config.enable_wandb else None
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