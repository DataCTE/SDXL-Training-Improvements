"""Logging utilities for SDXL training."""

# Base logging components first
from .config import LogConfig
from .base import LogManager, Logger, get_logger
from .utils import EnhancedFormatter, create_enhanced_logger, TensorLogger
from .setup import setup_logging

# Avoid circular imports by using __all__ first
__all__ = [
    "LogConfig",
    "LogManager",
    "Logger", 
    "get_logger",
    "MetricsLogger",
    "TrainingMetrics",
    "log_metrics",
    "WandbLogger",
    "EnhancedFormatter",
    "create_enhanced_logger",
    "TensorLogger",
    "setup_logging"
]

# Then import specialized loggers
from .metrics import MetricsLogger, TrainingMetrics, log_metrics
from .wandb import WandbLogger
