"""Centralized logging configuration for SDXL training."""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import threading
from datetime import datetime
import colorama
from .core import UnifiedLogger
from .base import LogConfig, ConfigurationError
from ..utils.paths import convert_windows_path

# Initialize colorama for Windows support
colorama.init(autoreset=True)

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _loggers: Dict[str, UnifiedLogger] = {}
    _lock = threading.Lock()
    _config = None
    
    def __init__(self):
        """Initialize log manager."""
        pass
        
    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_logger(self, name: str) -> UnifiedLogger:
        """Get or create UnifiedLogger by name."""
        with self._lock:
            if name not in self._loggers:
                config = self._config.copy() if self._config else LogConfig()
                config.name = name
                self._loggers[name] = UnifiedLogger(config)
            return self._loggers[name]
    
    def configure_from_config(self, config: Union[LogConfig, Dict]) -> None:
        """Configure logging from config object or dict."""
        if isinstance(config, dict):
            config = LogConfig(**config)
            
        self._config = config
        
        # Update existing loggers with new config
        with self._lock:
            for name, logger in self._loggers.items():
                new_config = config.copy()
                new_config.name = name
                self._loggers[name] = UnifiedLogger(new_config)
                
    def cleanup(self) -> None:
        """Cleanup all loggers."""
        with self._lock:
            for logger in self._loggers.values():
                logger.finish()
            self._loggers.clear()
            self._config = None

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
        config.log_dir = Path(convert_windows_path(log_dir, make_absolute=True))
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