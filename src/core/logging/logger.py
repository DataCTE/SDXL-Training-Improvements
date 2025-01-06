"""Unified logging system with configurable features."""
import logging
import sys
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from tqdm import tqdm
import torch
import wandb
import colorama
from colorama import Fore, Style

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
        self._metrics: Dict[str, deque] = {}
        self._progress = None
        self._wandb_run = None
        self._lock = threading.Lock()
        
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
            
    def _get_formatter(self, colored: bool = True) -> logging.Formatter:
        """Get log formatter with optional colors."""
        if not colored:
            return logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.MAGENTA
            }
            
            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
                return super().format(record)
                
        return ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
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
            self._progress = tqdm(
                total=total,
                desc=desc,
                smoothing=self.config.progress_smoothing
            )
            
    def update_progress(self, n: int = 1) -> None:
        """Update progress tracker."""
        if self._progress:
            self._progress.update(n)
            
            if self.config.enable_metrics:
                self.log_metric("progress/steps_per_sec", self._progress.format_dict["rate"])
                
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if self.config.enable_metrics:
            with self._lock:
                if name not in self._metrics:
                    self._metrics[name] = deque(maxlen=self.config.metrics_window)
                self._metrics[name].append(value)
                
                if self.config.enable_wandb and self._wandb_run:
                    self._wandb_run.log({name: value}, step=step)
                    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric averages."""
        return {
            name: float(np.mean(values))
            for name, values in self._metrics.items()
        }
        
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
