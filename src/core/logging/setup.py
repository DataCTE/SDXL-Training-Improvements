"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from .base import LogConfig, ColoredFormatter
from .utils import TensorLogger

def _configure_handlers(
    logger: logging.Logger,
    console_level: str = "INFO",
    file_path: Optional[Path] = None,
    file_level: str = "DEBUG"
) -> None:
    """Configure handlers for a logger with consistent formatting."""
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # Add file handler if path provided
    if file_path:
        file_handler = logging.FileHandler(
            file_path,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | '
            '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
            '%(funcName)s |\n%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

def setup_logging(
    config: Optional[Union[LogConfig, Dict]] = None,
    log_dir: Optional[str] = "outputs/logs", 
    level: Optional[Union[int, str]] = None,
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: Optional[bool] = None,
    propagate: Optional[bool] = None,
    console_level: Optional[Union[int, str]] = None
) -> Tuple[logging.Logger, TensorLogger]:
    """Setup logging system with configuration."""
    # Get root logger
    root_logger = logging.getLogger()
    
    # Set root logger to DEBUG to allow all messages through
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, (console_level or "INFO").upper()))
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # Add file handler if filename provided
    if filename:
        file_handler = logging.FileHandler(
            Path(log_dir) / filename,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, (level or "DEBUG").upper()))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | '
            '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
            '%(funcName)s |\n%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)
        
    # Configure warning capture
    logging.captureWarnings(capture_warnings if capture_warnings is not None else True)
    
    # Create and return loggers
    logger = logging.getLogger(module_name or 'root')
    tensor_logger = TensorLogger(logger)
    
    return logger, tensor_logger
