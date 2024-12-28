"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from .base import LogConfig, ColoredFormatter
from .utils import TensorLogger

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
"""Logging setup utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from .base import LogConfig, ColoredFormatter
from .utils import TensorLogger

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
    """Setup logging system with configuration.
    
    Args:
        config: LogConfig object or dict containing logging configuration
        log_dir: Directory for log files (fallback if not in config)
        level: Logging level for file output (fallback if not in config)
        filename: Optional log file name (fallback if not in config)
        module_name: Optional module name for logger
        capture_warnings: Whether to capture Python warnings (fallback if not in config)
        propagate: Whether to propagate logs to parent loggers (fallback if not in config)
        console_level: Logging level for console output (fallback if not in config)
        
    Returns:
        Tuple of (configured logger instance, tensor logger instance)
    """
    # Use config values if provided, otherwise use parameters
    if config:
        if isinstance(config, dict):
            config = LogConfig(**config)
        log_dir = config.log_dir
        level = config.file_level
        filename = config.filename
        capture_warnings = config.capture_warnings
        propagate = config.propagate
        console_level = config.console_level

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
        file_path = Path(log_dir) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            file_path,
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
    
    # Set propagate flag if specified
    if propagate is not None:
        logger.propagate = propagate
    
    tensor_logger = TensorLogger(logger)
    
    return logger, tensor_logger
"""Logging setup and configuration."""
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from .base import LogConfig
from .utils import EnhancedFormatter, TensorLogger

# Global logger registry to prevent duplicates
_logger_registry = {}

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
    global _logger_registry
    
    # If logger already exists for this module, return it
    if module_name in _logger_registry:
        return _logger_registry[module_name]
    
    # Configure root logger first
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all messages through
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.propagate = False  # Prevent propagation to avoid duplicates
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, (console_level or "INFO").upper()))
    console_handler.setFormatter(EnhancedFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # Add file handler if filename provided
    if filename:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / filename,
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
    
    # Create module logger
    logger = logging.getLogger(module_name or 'root')
    logger.propagate = False if propagate is None else propagate
    logger.setLevel(logging.DEBUG)  # Allow all messages through
    
    # Create tensor logger
    tensor_logger = TensorLogger(logger, dump_on_error=True)
    
    # Store in registry
    _logger_registry[module_name or 'root'] = (logger, tensor_logger)
    
    # Suppress specific loggers that might cause duplicates
    for logger_name in ["PIL", "torch", "transformers", "diffusers"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).propagate = False
    
    return logger, tensor_logger
