"""Centralized logging system for SDXL training."""
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
from dataclasses import dataclass
import wandb
from PIL import Image
import colorama
from colorama import Fore, Style
from datetime import datetime
from functools import wraps
from ..distributed import reduce_dict

# Initialize colorama
colorama.init(autoreset=True)

@dataclass
class LogConfig:
    """Unified logging configuration."""
    # Basic logging config
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "outputs/logs"
    filename: str = "train.log"
    capture_warnings: bool = True
    console_output: bool = True
    file_output: bool = True
    
    # Metrics config
    metrics_window_size: int = 100
    log_cuda_memory: bool = True
    log_system_memory: bool = True
    performance_logging: bool = True
    
    # W&B config
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = None
    wandb_notes: str = None
    
    def __post_init__(self):
        self.wandb_tags = self.wandb_tags or []

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _loggers: Dict[str, 'Logger'] = {}
    _metrics_buffer: Dict[str, List[float]] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'LogManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_logger(self, name: str, config: LogConfig) -> 'Logger':
        with self._lock:
            if name not in self._loggers:
                self._loggers[name] = Logger(name, config)
            return self._loggers[name]

class Logger:
    """Unified logger combining console, file, metrics, and W&B logging."""
    
    def __init__(self, name: str, config: LogConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.metrics_buffer = {}
        self.wandb_run = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Initialize all logging components."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set base level
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_output:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_path / self.config.filename,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | '
                '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
                '%(funcName)s |\n%(message)s\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            file_handler.setLevel(getattr(logging, self.config.file_level.upper()))
            self.logger.addHandler(file_handler)
        
        # Initialize W&B if enabled
        if self.config.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_name,
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes
                )
            except Exception as e:
                self.error(f"Failed to initialize W&B: {str(e)}", exc_info=True)
                self.wandb_run = None

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
        reduce: bool = True
    ) -> None:
        """Log metrics to all enabled outputs."""
        try:
            # Reduce metrics across processes if needed
            if reduce:
                metrics = reduce_dict(metrics)
            
            # Update metrics buffer
            for k, v in metrics.items():
                if k not in self.metrics_buffer:
                    self.metrics_buffer[k] = []
                self.metrics_buffer[k].append(float(v))
                if len(self.metrics_buffer[k]) > self.config.metrics_window_size:
                    self.metrics_buffer[k].pop(0)
            
            # Log to console
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            self.info(f"Step {step}: {metrics_str}")
            
            # Log to W&B
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=step, commit=commit)
                
        except Exception as e:
            self.error(f"Failed to log metrics: {str(e)}", exc_info=True)

    def log_images(
        self,
        images: Dict[str, Union[Image.Image, torch.Tensor]],
        step: Optional[int] = None
    ) -> None:
        """Log images to W&B and/or filesystem."""
        try:
            # Save to filesystem
            if self.config.file_output:
                img_path = Path(self.config.log_dir) / "images"
                img_path.mkdir(parents=True, exist_ok=True)
                for name, img in images.items():
                    if isinstance(img, torch.Tensor):
                        img = self._tensor_to_pil(img)
                    img.save(img_path / f"{name}_{step}.png")
            
            # Log to W&B
            if self.wandb_run is not None:
                wandb_images = {
                    name: wandb.Image(img if isinstance(img, Image.Image) else self._tensor_to_pil(img))
                    for name, img in images.items()
                }
                self.wandb_run.log(wandb_images, step=step)
                
        except Exception as e:
            self.error(f"Failed to log images: {str(e)}", exc_info=True)

    def log_model(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None
    ) -> None:
        """Log model architecture and gradients."""
        try:
            if self.wandb_run is not None:
                # Log model architecture
                self.wandb_run.watch(
                    model,
                    log="all",
                    log_freq=100,
                    log_graph=True
                )
                
                # Log parameter statistics
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float('inf')
                )
                
                self.log_metrics({
                    "model/gradient_norm": grad_norm.item(),
                    "model/parameter_norm": sum(
                        p.norm().item() ** 2 
                        for p in model.parameters()
                    ) ** 0.5
                }, step=step)
                
        except Exception as e:
            self.error(f"Failed to log model: {str(e)}", exc_info=True)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.ndim == 4:
            tensor = tensor[0]
        tensor = (tensor * 255).byte().cpu().numpy()
        if tensor.shape[0] == 3:
            tensor = tensor.transpose(1, 2, 0)
        return Image.fromarray(tensor)

    # Delegate basic logging methods
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

def get_logger(name: str, config: Optional[LogConfig] = None) -> Logger:
    """Get or create a logger instance."""
    if config is None:
        config = LogConfig()
    return LogManager.get_instance().get_logger(name, config)