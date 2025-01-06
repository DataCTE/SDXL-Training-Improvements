"""Unified logging system for SDXL training."""
from typing import TypeAlias
from .base import LogConfig, ProgressConfig, MetricsConfig
from .core import UnifiedLogger
from .logging import setup_logging, cleanup_logging, LogManager, get_logger
from .formatters import ColoredFormatter
from .progress import ProgressTracker
from .metrics import MetricsTracker

# Type alias for backward compatibility
Logger: TypeAlias = UnifiedLogger

__all__ = [
    'LogConfig',
    'ProgressConfig', 
    'MetricsConfig',
    'UnifiedLogger',
    'Logger',  # Backward compatibility
    'setup_logging',
    'cleanup_logging',
    'LogManager',
    'ColoredFormatter',
    'ProgressTracker',
    'MetricsTracker',
    'get_logger'
]
