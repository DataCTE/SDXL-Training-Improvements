"""Unified logging system for SDXL training."""
from typing import Optional, Union, Dict, Any
import logging
from pathlib import Path

from .base import LogConfig, ConfigurationError, MetricsConfig, ProgressConfig
from .core import UnifiedLogger, LogManager
from .wandb import WandbLogger, WandbConfig
from .formatters import ColoredFormatter
from .metrics import MetricsTracker
from .progress import ProgressTracker
from .progress_predictor import ProgressPredictor

def setup_logging(
    config: Optional[Union[LogConfig, Dict]] = None,
    log_dir: Optional[str] = None,
    level: Optional[Union[int, str]] = None,
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: Optional[bool] = None,
    propagate: Optional[bool] = None,
    console_level: Optional[Union[int, str]] = None,
    enable_wandb: Optional[bool] = None,
    enable_progress: Optional[bool] = None,
    enable_metrics: Optional[bool] = None
) -> UnifiedLogger:
    """Setup logging system with enhanced configuration.
    
    Args:
        config: Base configuration object or dict
        log_dir: Override log directory
        level: Override file logging level
        filename: Override log filename
        module_name: Override logger name
        capture_warnings: Override warning capture
        propagate: Override log propagation
        console_level: Override console logging level
        enable_wandb: Override W&B logging
        enable_progress: Override progress tracking
        enable_metrics: Override metrics tracking
        
    Returns:
        Configured UnifiedLogger instance
    """
    # Create config from parameters or use provided config
    if config is None:
        config = LogConfig()
        
    if isinstance(config, dict):
        config = LogConfig(**config)
        
    # Override config with explicit parameters if provided
    if log_dir is not None:
        config.log_dir = Path(log_dir)
    if level is not None:
        config.file_level = level
    if filename is not None:
        config.filename = filename
    if capture_warnings is not None:
        config.capture_warnings = capture_warnings
    if propagate is not None:
        config.propagate = propagate
    if console_level is not None:
        config.console_level = console_level
    if module_name is not None:
        config.name = module_name
    if enable_wandb is not None:
        config.enable_wandb = enable_wandb
    if enable_progress is not None:
        config.enable_progress = enable_progress
    if enable_metrics is not None:
        config.enable_metrics = enable_metrics
        
    # Configure warning capture
    logging.captureWarnings(config.capture_warnings)
    
    # Get logger from manager
    manager = LogManager.get_instance()
    manager.configure_from_config(config)
    return manager.get_logger(config.name)
    
def cleanup_logging() -> None:
    """Cleanup logging system."""
    manager = LogManager.get_instance()
    manager.cleanup()
    
    # Clean up root logger handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except Exception as e:
            logging.error(f"Error closing handler: {str(e)}", exc_info=True)
    
    logging.info("Logging system cleanup complete")

def get_logger(name: str) -> UnifiedLogger:
    """Get a logger by name with enhanced features.
    
    This provides a unified interface for logging with support for:
    - Colored console output
    - File logging
    - W&B integration
    - Progress tracking
    - Metrics monitoring
    - Memory tracking
    
    Args:
        name: Logger name
        
    Returns:
        UnifiedLogger instance with all configured features
    """
    return LogManager.get_instance().get_logger(name)

__all__ = [
    # Core components
    'LogConfig',
    'WandbConfig',
    'MetricsConfig',
    'ProgressConfig',
    'UnifiedLogger',
    'LogManager',
    
    # Feature-specific
    'WandbLogger',
    'ColoredFormatter',
    'MetricsTracker',
    'ProgressTracker',
    'ProgressPredictor',
    
    # Exceptions
    'ConfigurationError',
    
    # Helper functions
    'setup_logging',
    'cleanup_logging',
    'get_logger'
]
