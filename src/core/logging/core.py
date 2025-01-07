"""Core logging functionality with integrated features."""
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import torch
import wandb
from tqdm import tqdm
from .base import LogConfig, ProgressConfig, MetricsConfig, ConfigurationError
from .formatters import ColoredFormatter
from .progress import ProgressTracker
from .metrics import MetricsTracker
from .progress_predictor import ProgressPredictor
from ..utils.paths import convert_windows_path

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _loggers: Dict[str, 'UnifiedLogger'] = {}
    _lock = threading.Lock()
    _config = None
    
    def __init__(self):
        """Initialize log manager."""
        pass
        
    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_logger(self, name: str) -> 'UnifiedLogger':
        """Get or create UnifiedLogger by name."""
        with self._lock:
            if name not in self._loggers:
                config = self._config.copy() if self._config else LogConfig()
                config.name = name
                self._loggers[name] = UnifiedLogger(config)
            return self._loggers[name]
    
    def configure_from_config(self, config: Union[LogConfig, Dict]) -> None:
        """Configure logging from config object or dict."""
        if isinstance(config, dict):
            config = LogConfig(**config)
            
        self._config = config
        
        # Update existing loggers with new config
        with self._lock:
            for name, logger in self._loggers.items():
                new_config = config.copy()
                new_config.name = name
                self._loggers[name] = UnifiedLogger(new_config)
                
    def cleanup(self) -> None:
        """Cleanup all loggers."""
        with self._lock:
            for logger in self._loggers.values():
                logger.finish()
            self._loggers.clear()
            self._config = None

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
            log_dir = Path(convert_windows_path(str(self.config.log_dir), make_absolute=True))
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
        # W&B initialization with enhanced error handling
        self._wandb_run = None
        if self.config.enable_wandb:
            try:
                self._wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_name,
                    tags=self.config.wandb_tags,
                    config=vars(self.config),
                    resume=True,  # Enable run resumption
                    settings=wandb.Settings(
                        start_method="thread",
                        _disable_stats=True
                    )
                )
                self.logger.info("Initialized W&B logging", extra={
                    'project': self.config.wandb_project,
                    'run_name': self.config.wandb_name,
                    'tags': self.config.wandb_tags,
                    'run_id': self._wandb_run.id,
                    'run_url': self._wandb_run.get_url()
                })
            except Exception as e:
                self.logger.warning(
                    "Failed to initialize W&B logging. Continuing without W&B.", 
                    extra={'error': str(e)}, 
                    exc_info=True
                )
                
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
            
        # Progress tracking setup with memory tracking
        self._progress = None
        self._predictor = None
        self._progress_bar = None
        self._completed_items = 0
        self._total_items = 0
        self._last_memory_check = 0
        self._memory_check_interval = 10  # Check memory every 10 updates
        
    def start_progress(self, total: int, desc: str = "") -> None:
        """Start progress tracking with smart ETA prediction."""
        if self.config.enable_progress:
            # Initialize predictor
            self._predictor = ProgressPredictor()
            self._predictor.start(total)
            
            # Initialize progress bar
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                leave=self.config.progress_leave,
                position=self.config.progress_position,
                dynamic_ncols=True
            )
            
            # Initialize standard progress tracker
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
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> None:
        """Log multiple metrics with optional step and context.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional global step number
            context: Optional context dictionary for additional logging info
        """
        if not self.config.enable_metrics:
            return
            
        for name, value in metrics.items():
            self._metrics.update(name, value)
            
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)
            
        # Log with context if provided
        if context:
            self.logger.info("Metrics update", extra={
                'metrics': metrics,
                'step': step,
                **context
            })
            
    def update_progress(
        self,
        n: int = 1,
        batch_size: Optional[int] = None,
        prefix: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, float]]:
        """Update progress with smart ETA prediction and enhanced logging.
        
        Args:
            n: Number of steps completed
            batch_size: Optional batch size for throughput calculation
            prefix: Optional prefix for progress description
            context: Optional context dictionary for additional logging info
            
        Returns:
            Dictionary with progress metrics if available
        """
        if not self.config.enable_progress:
            return None
            
        # Update standard progress metrics
        metrics = None
        if self._progress is not None:
            metrics = self._progress.update(n, batch_size)
            if self.config.enable_metrics and metrics:
                self.log_metrics(metrics)
                
        # Update predictor and get timing info
        if self._predictor is not None:
            timing = self._predictor.update(n)
            current_time = timing["current_time"]
            eta = timing["eta_seconds"]
            
            # Format timing for display
            iter_time = f"{current_time*1000:.1f}ms/it"
            eta_str = f"ETA: {self._predictor.format_time(eta)}"
            
            # Update progress bar with enhanced info
            if self._progress_bar is not None:
                desc = f"{prefix + ': ' if prefix else ''}"
                status = f"{iter_time} • {eta_str}"
                if context:
                    extra_info = " • ".join(f"{k}: {v}" for k, v in context.items())
                    status = f"{status} • {extra_info}"
                self._progress_bar.set_postfix_str(status)
                self._progress_bar.update(n)
                
                # Log detailed progress at intervals
                if self._completed_items % max(100, self._total_items // 20) == 0:
                    progress_pct = (self._completed_items / self._total_items * 100) if self._total_items > 0 else 0
                    self.logger.info(f"Progress update: {progress_pct:.1f}%", extra={
                        'completed': self._completed_items,
                        'total': self._total_items,
                        'progress_percent': progress_pct,
                        'eta_seconds': eta,
                        'iter_time': current_time,
                        **(context or {})
                    })
                
            # Add timing and memory metrics
            if metrics is None:
                metrics = {}
            metrics.update({
                "progress/iter_time": current_time,
                "progress/eta_seconds": eta,
                "progress/percent_complete": progress
            })
            
            # Periodic memory tracking
            if self.config.enable_memory and torch.cuda.is_available():
                if self._completed_items % self._memory_check_interval == 0:
                    metrics.update({
                        "memory/gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "memory/gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                        "memory/gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
                    })
            
        return metrics
            
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
            
        if self._progress_bar is not None:
            self._progress_bar.close()
            
        self._predictor = None
            
    def __enter__(self) -> 'UnifiedLogger':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.finish()
