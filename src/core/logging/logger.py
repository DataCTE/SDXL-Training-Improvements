"""Unified logging system with configurable features."""
import logging
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from collections import deque
import numpy as np
from tqdm import tqdm
import torch
import wandb
import colorama
from colorama import Fore, Style
from .base import LogConfig

# Initialize colorama
colorama.init(autoreset=True)

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

class UnifiedLogger:
    """Centralized logging system with optional features."""
    
    def __init__(self, config: Optional[Union[LogConfig, Dict]] = None):
        """Initialize logger with configuration."""
        self.config = LogConfig(**config) if isinstance(config, dict) else config or LogConfig()
        self._setup_logging()
        self._metrics = MetricsTracker(
            window_size=self.config.metrics_window,
            keep_history=self.config.metrics_history
        )
        self._progress = None
        self._wandb_run = None
        
    def _setup_logging(self):
        """Configure logging handlers."""
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        if self.config.enable_console:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(getattr(logging, self.config.console_level))
            console.setFormatter(self._get_formatter())
            self.logger.addHandler(console)
            
        if self.config.enable_file:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
            file = logging.FileHandler(
                self.config.log_dir / self.config.filename,
                encoding='utf-8'
            )
            file.setLevel(getattr(logging, self.config.file_level))
            file.setFormatter(self._get_formatter(colored=False))
            self.logger.addHandler(file)
            
        if self.config.enable_wandb:
            self._setup_wandb()
            
class ColoredFormatter(logging.Formatter):
    """Enhanced formatter with colored output and context tracking."""
    
    COLORS = {
        'DEBUG': Fore.CYAN + Style.DIM,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW + Style.BRIGHT,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT + Style.DIM
    }

    HIGHLIGHT_COLORS = {
        'file_path': Fore.BLUE,
        'line_number': Fore.CYAN,
        'function': Fore.MAGENTA,
        'error': Fore.RED + Style.BRIGHT,
        'success': Fore.GREEN + Style.BRIGHT,
        'warning': Fore.YELLOW + Style.BRIGHT
    }

    KEYWORDS = {
        'start': (Fore.CYAN, ['Starting', 'Initializing', 'Beginning']),
        'success': (Fore.GREEN, ['Complete', 'Finished', 'Saved', 'Success']),
        'error': (Fore.RED, ['Error', 'Failed', 'Exception']),
        'warning': (Fore.YELLOW, ['Warning', 'Caution']),
        'progress': (Fore.BLUE, ['Processing', 'Loading', 'Computing'])
    }

    def __init__(self, fmt=None, datefmt=None, colored=True):
        super().__init__(fmt or '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                        datefmt or '%Y-%m-%d %H:%M:%S')
        self.colored = colored

    def format(self, record):
        if not self.colored:
            return super().format(record)
            
        # Create a copy of the record
        filtered_record = logging.makeLogRecord(record.__dict__)
        
        # Get base color for level
        base_color = self.COLORS.get(record.levelname, '')
        
        # Format with parent
        formatted_message = super().format(filtered_record)
        
        # Apply keyword highlighting
        for keyword, (color, words) in self.KEYWORDS.items():
            for word in words:
                if word in formatted_message:
                    formatted_message = formatted_message.replace(
                        word, f"{color}{word}{Style.RESET_ALL}")
                    setattr(record, 'keyword', keyword)
        
        # Apply context highlighting
        for context, color in self.HIGHLIGHT_COLORS.items():
            if hasattr(record, context):
                value = getattr(record, context)
                formatted_message = formatted_message.replace(
                    str(value), f"{color}{value}{Style.RESET_ALL}")
        
        return f"{base_color}{formatted_message}{Style.RESET_ALL}"

class ProgressTracker:
    """Tracks progress with throughput monitoring."""
    
    def __init__(self, total: int, desc: str = "", 
                 window_size: int = 100, smoothing: float = 0.3,
                 metric_prefix: str = "throughput/"):
        self.progress = tqdm(total=total, desc=desc, smoothing=smoothing)
        self.window_size = window_size
        self.steps = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.last_update = time.time()
        self.metric_prefix = metric_prefix
        self._total_samples = 0
        self._last_metrics = {}
        
    def update(self, n: int = 1, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Update progress and return metrics."""
        self.progress.update(n)
        current = time.time()
        
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
            self._total_samples += batch_size
            
        elapsed = current - self.last_update
        self.steps.append(elapsed / n)
        self.last_update = current
        
        return self.get_metrics()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current throughput metrics."""
        if not self.steps:
            return self._last_metrics
            
        avg_time = sum(self.steps) / len(self.steps)
        total_samples = sum(self.batch_sizes) if self.batch_sizes else self._total_samples
        samples_per_sec = (
            total_samples / sum(self.steps) if self.batch_sizes 
            else self._total_samples / sum(self.steps)
        )
        
        metrics = {
            f"{self.metric_prefix}samples_per_sec": samples_per_sec,
            f"{self.metric_prefix}batch_time_ms": avg_time * 1000,
            f"{self.metric_prefix}accumulated_samples": total_samples
        }
        self._last_metrics = metrics
        return metrics
        
    def close(self):
        """Clean up progress bar."""
        self.progress.close()

class MetricsTracker:
    """Tracks metrics with configurable window size and history."""
    
    def __init__(self, window_size: int = 100, keep_history: bool = True):
        self.window_size = window_size
        self.keep_history = keep_history
        self.metrics: Dict[str, deque] = {}
        self.history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        
    def update(self, name: str, value: float) -> None:
        """Update metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
                if self.keep_history:
                    self.history[name] = []
                    
            self.metrics[name].append(value)
            if self.keep_history:
                self.history[name].append(value)
                
    def get_average(self, name: str) -> Optional[float]:
        """Get current average for metric."""
        values = self.metrics.get(name)
        if values:
            return float(np.mean(values))
        return None
        
    def get_all_averages(self) -> Dict[str, float]:
        """Get current averages for all metrics."""
        return {
            name: float(np.mean(values))
            for name, values in self.metrics.items()
        }
        
    def get_history(self, name: str) -> Optional[List[float]]:
        """Get full history for metric if available."""
        return self.history.get(name)

    def _get_formatter(self, colored: bool = True) -> logging.Formatter:
        """Get log formatter with optional colors."""
        return ColoredFormatter(colored=colored)
        
    def _setup_wandb(self):
        """Initialize W&B logging."""
        try:
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                tags=self.config.wandb_tags,
                config=vars(self.config)
            )
        except Exception as e:
            self.warning(f"Failed to initialize W&B: {e}")
            
    def start_progress(self, total: int, desc: str = "") -> None:
        """Start progress tracking."""
        if self.config.enable_progress:
            self._progress = ProgressTracker(
                total=total,
                desc=desc,
                window_size=self.config.progress_window,
                smoothing=self.config.progress_smoothing
            )
            
    def update_progress(self, n: int = 1) -> None:
        """Update progress tracker."""
        if self._progress:
            rate = self._progress.update(n)
            if self.config.enable_metrics:
                self.log_metric("progress/steps_per_sec", rate)
                
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if self.config.enable_metrics:
            self._metrics.update(name, value)
            if self.config.enable_wandb and self._wandb_run:
                self._wandb_run.log({name: value}, step=step)
                    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric averages."""
        return self._metrics.get_all_averages()
        
    def log_memory(self) -> None:
        """Log memory usage metrics."""
        if self.config.enable_memory and torch.cuda.is_available():
            self.log_metric("memory/gpu_allocated_gb", 
                          torch.cuda.memory_allocated() / 1e9)
            self.log_metric("memory/gpu_reserved_gb",
                          torch.cuda.memory_reserved() / 1e9)
            
    # Standard logging methods
    def debug(self, msg: str, *args, **kwargs): 
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs): 
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs): 
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs): 
        self.logger.error(msg, *args, **kwargs)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.close()
        if self._wandb_run:
            self._wandb_run.finish()
