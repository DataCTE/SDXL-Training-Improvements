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

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class LogConfig:
    """Unified logging configuration."""
    # Core settings
    name: str = "root"
    log_dir: Path = Path("outputs/logs")
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    filename: str = "training.log"
    
    # Feature flags
    enable_console: bool = True
    enable_file: bool = True
    enable_wandb: bool = False
    enable_progress: bool = True
    enable_metrics: bool = True
    enable_memory: bool = True
    capture_warnings: bool = True
    propagate: bool = False
    
    # W&B settings
    wandb_project: str = "default"
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Progress settings
    progress_window: int = 100
    progress_smoothing: float = 0.3
    
    # Metrics settings
    metrics_window: int = 100
    metrics_history: bool = True

    def copy(self) -> 'LogConfig':
        """Create a copy of the config."""
        return LogConfig(**self.__dict__)

from .logger import ColoredFormatter, ProgressTracker, MetricsTracker

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
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log multiple metrics with optional step."""
        for name, value in metrics.items():
            self._metrics.update(name, value)
            
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
            
    def update_progress(self, n: int = 1, **kwargs) -> Optional[float]:
        """Update progress and return current rate if available."""
        if self._progress is not None:
            rate = self._progress.update(n)
            if self.config.enable_metrics:
                self._metrics.update("progress/steps_per_sec", rate)
            return rate
        return None
            
    def finish(self) -> None:
        """Cleanup and close all logging systems."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            
        if self._progress is not None:
            self._progress.close()
            
    def __enter__(self) -> 'UnifiedLogger':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.finish()
