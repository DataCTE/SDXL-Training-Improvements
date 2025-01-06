"""Centralized logging configuration for SDXL training."""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import threading
from datetime import datetime
import colorama
from .logger import ColoredFormatter, UnifiedLogger
from .base import LogConfig
from src.data.utils.paths import convert_windows_path

# Initialize colorama for Windows support
colorama.init(autoreset=True)

class TensorLogger:
    """Logger for tensor operations."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

# Global action history dict
_action_history: Dict[str, Any] = {}

from .base import LogConfig

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _tensor_loggers: Dict[str, TensorLogger] = {}
    _action_history: Dict[str, Any] = {}
    
    def __init__(self):
        self._lock = threading.Lock()
        
    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create logger by name."""
        with self._lock:
            if name not in self._loggers:
                self._loggers[name] = logging.getLogger(name)
            return self._loggers[name]
    
    def get_tensor_logger(self, name: str) -> TensorLogger:
        """Get or create tensor logger by name."""
        with self._lock:
            if name not in self._tensor_loggers:
                self._tensor_loggers[name] = TensorLogger(self.get_logger(name))
            return self._tensor_loggers[name]
    
    def configure_from_config(self, config: Union[LogConfig, Dict]) -> None:
        """Configure logging from config object or dict."""
        if isinstance(config, dict):
            config = LogConfig(**config)
            
        # Create log directory
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(config.get_file_level())
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler if enabled
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.get_console_level())
            console_handler.setFormatter(ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if config.file_output and config.filename:
            file_handler = logging.FileHandler(
                log_dir / config.filename,
                mode='a',
                encoding='utf-8'
            )
            file_handler.setLevel(config.get_file_level())
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | '
                '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
                '%(funcName)s |\n%(message)s\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            root_logger.addHandler(file_handler)
        
        # Configure warning capture
        logging.captureWarnings(config.capture_warnings)
        
        # Store config
        self.config = config


    

def setup_logging(
    config: Optional[Union[LogConfig, Dict]] = None,
    log_dir: Optional[str] = "outputs/logs", 
    level: Optional[Union[int, str]] = None,
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: Optional[bool] = None,
    propagate: Optional[bool] = None,
    console_level: Optional[Union[int, str]] = None
) -> Tuple[logging.Logger, "TensorLogger"]:
    """Setup logging system with configuration."""
    # Use config values if provided, otherwise use fallback values
    if config:
        if isinstance(config, dict):
            config = LogConfig(**config)
        log_dir = config.log_dir
        level = config.file_level
        filename = config.filename
        capture_warnings = config.capture_warnings
        propagate = config.propagate
        console_level = config.console_level

    # Create log directory only for file logging
    if filename:
        log_path = Path(convert_windows_path(log_dir, make_absolute=True))
        log_path.mkdir(parents=True, exist_ok=True)
        
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all messages through
    root_logger.handlers.clear()
    
    # Add console handler with specified level
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

    # Configure warning capture and propagation
    logging.captureWarnings(capture_warnings if capture_warnings is not None else True)
    
    # Create logger with propagation explicitly set
    logger = logging.getLogger(module_name or 'root')
    logger.propagate = False if propagate is None else propagate
    
    # Create tensor logger
    tensor_logger = TensorLogger(logger)
    
    return logger, tensor_logger
    
def cleanup_logging() -> Dict[str, Any]:
    """Cleanup logging handlers and return action history.
    
    Returns:
        Dictionary containing logged action history
    """
    global logger
    logger = logging.getLogger(__name__)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except Exception as e:
            logger.error(f"Error closing handler: {str(e)}", exc_info=True, stack_info=True)
    
    # Log cleanup
    logging.info("Logging system cleanup complete")
    
    # Return action history for analysis
    return {
        'actions': _action_history,
        'total_actions': len(_action_history),
        'categories': {
            category: len([a for a in _action_history.values() if a['category'] == category])
            for category in set(a['category'] for a in _action_history.values())
        }
    }
