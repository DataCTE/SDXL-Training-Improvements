"""Unified logging system for SDXL training."""
import logging

# Basic logger setup
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

# Import after get_logger to avoid circular imports
from .base import LogConfig
from .logger import ColoredFormatter, ProgressTracker, MetricsTracker, UnifiedLogger
from .logging import LogManager, setup_logging, cleanup_logging

__all__ = [
    "LogConfig",
    "UnifiedLogger",
    "ColoredFormatter", 
    "ProgressTracker",
    "MetricsTracker",
    "LogManager",
    "setup_logging",
    "cleanup_logging",
    "get_logger"
]
