"""Centralized logging configuration for SDXL training."""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import threading
import colorama
from colorama import Fore, Style
from datetime import datetime
from .base import LogConfig
from .utils import ColoredFormatter, TensorLogger
from src.data.utils.paths import convert_windows_path

# Initialize colorama for Windows support
colorama.init(autoreset=True)

# Global action history dict
_action_history: Dict[str, Any] = {}

from .base import LogConfig

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _tensor_loggers: Dict[str, 'TensorLogger'] = {}
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
    
    def get_tensor_logger(self, name: str) -> 'TensorLogger':
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


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output with detailed context."""
    
    COLORS = {
        'DEBUG': Fore.CYAN + Style.DIM,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW + Style.BRIGHT,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT + Style.DIM
    }

    HIGHLIGHT_COLORS = {
        'file_path': Fore.BLUE,
        'line_number': Fore.CYAN,
        'function': Fore.MAGENTA,
        'error': Fore.RED + Style.BRIGHT,
        'success': Fore.GREEN + Style.BRIGHT,
        'warning': Fore.YELLOW + Style.BRIGHT
    }

    KEYWORDS = {
        'start': (Fore.CYAN, ['Starting', 'Initializing', 'Beginning']),
        'success': (Fore.GREEN, ['Complete', 'Finished', 'Saved', 'Success']),
        'error': (Fore.RED, ['Error', 'Failed', 'Exception']),
        'warning': (Fore.YELLOW, ['Warning', 'Caution']),
        'progress': (Fore.BLUE, ['Processing', 'Loading', 'Computing'])
    }

    def format(self, record):
        """Enhanced formatter with detailed context and error tracking."""
        # Create a copy of the record to avoid modifying the original
        filtered_record = logging.makeLogRecord(record.__dict__)
        
        # Get timestamp once for consistency
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Get the base color for the entire message based on level
        base_color = self.COLORS.get(record.levelname, '')
        
        # Format the record using the parent formatter
        formatted_message = super().format(filtered_record)
        
        # Apply keyword highlighting
        for keyword, (color, words) in self.KEYWORDS.items():
            for word in words:
                if word in formatted_message:
                    formatted_message = formatted_message.replace(word, f"{color}{word}{Style.RESET_ALL}")
                    # Add keyword to the record for context
                    setattr(record, 'keyword', keyword)
        
        # Apply context highlighting
        for context, color in self.HIGHLIGHT_COLORS.items():
            if hasattr(record, context):
                value = getattr(record, context)
                formatted_message = formatted_message.replace(str(value), f"{color}{value}{Style.RESET_ALL}")
        
        # Add timestamp to the formatted message
        formatted_message = f"{Fore.WHITE}{timestamp}{Style.RESET_ALL} | {formatted_message}"
        
        # Apply the base color to the entire formatted message
        colored_message = f"{base_color}{formatted_message}{Style.RESET_ALL}"
        
        return colored_message
    
   
    

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
