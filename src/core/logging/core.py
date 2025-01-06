"""Core logging functionality with integrated features."""
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import torch
import wandb
from .base import LogConfig, ProgressConfig, MetricsConfig, ConfigurationError
from .formatters import ColoredFormatter
from .progress import ProgressTracker
from .metrics import MetricsTracker

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
        self.logger.propagate = self.config.propagate
        
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
                
        # Metrics tracking setup
        if self.config.enable_metrics:
            metrics_config = MetricsConfig(
                window_size=self.config.metrics_window,
                keep_history=self.config.metrics_history,
                prefix=self.config.metrics_prefix
            )
            self._metrics = MetricsTracker(metrics_config)
        else:
            self._metrics = None
            
        # Progress tracking setup
        self._progress = None
        
    def start_progress(self, total: int, desc: str = "") -> None:
        """Start progress tracking."""
        if self.config.enable_progress:
            progress_config = ProgressConfig(
                total=total,
                desc=desc,
                smoothing=self.config.progress_smoothing,
                position=self.config.progress_position,
                leave=self.config.progress_leave,
                window_size=self.config.progress_window
            )
            self._progress = ProgressTracker(progress_config)
    
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
        if not self.config.enable_metrics:
            return
            
        for name, value in metrics.items():
            self._metrics.update(name, value)
            
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
            
    def update_progress(self, n: int = 1, batch_size: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Update progress and return metrics if available."""
        if self._progress is not None:
            metrics = self._progress.update(n, batch_size)
            if self.config.enable_metrics and metrics:
                self.log_metrics(metrics)
            return metrics
        return None
            
    def log_memory(self) -> None:
        """Log memory usage metrics."""
        if not self.config.enable_memory or not torch.cuda.is_available():
            return
            
        self.log_metrics({
            "memory/gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory/gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9
        })
            
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
