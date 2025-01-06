
from typing import Optional, Union, Dict
import logging
from pathlib import Path

from .base import LogConfig, ConfigurationError
from .core import UnifiedLogger, LogManager
from .wandb import WandbLogger, WandbConfig

def setup_logging(
    config: Optional[Union[LogConfig, Dict]] = None,
    log_dir: Optional[str] = None,
    level: Optional[Union[int, str]] = None,
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: Optional[bool] = None,
    propagate: Optional[bool] = None,
    console_level: Optional[Union[int, str]] = None
) -> UnifiedLogger:
    """Setup logging system with configuration."""
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
    """Get a logger by name.
    
    This is a compatibility function that provides the same interface
    as the old get_logger function but uses the new UnifiedLogger system.
    
    Args:
        name: Logger name
        
    Returns:
        UnifiedLogger instance
    """
    return LogManager.get_instance().get_logger(name)

__all__ = [
    'LogConfig',
    'WandbConfig',
    'UnifiedLogger',
    'WandbLogger',
    'ConfigurationError',
    'setup_logging',
    'cleanup_logging',
    'get_logger'
]