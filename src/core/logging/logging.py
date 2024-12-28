"""Centralized logging configuration for SDXL training."""
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import colorama
from colorama import Fore, Style
from datetime import datetime
import threading
import torch
from dataclasses import dataclass

# Initialize colorama for Windows support
colorama.init(autoreset=True)

# Global action history dict
_action_history: Dict[str, Any] = {}

@dataclass
class LoggingConfig:
    """Centralized logging configuration."""
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "outputs/wslref/logs"
    filename: str = "train.log"
    capture_warnings: bool = True
    console_output: bool = True
    file_output: bool = True
    log_cuda_memory: bool = True
    log_system_memory: bool = True
    performance_logging: bool = True
    propagate: bool = True
    module_name: Optional[str] = None
    
    def get_console_level(self) -> int:
        """Convert string level to logging constant."""
        return getattr(logging, self.console_level.upper())
    
    def get_file_level(self) -> int:
        """Convert string level to logging constant."""
        return getattr(logging, self.file_level.upper())

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
    
    def configure_from_config(self, config: Union[LoggingConfig, Dict]) -> None:
        """Configure logging from config object or dict."""
        if isinstance(config, dict):
            config = LoggingConfig(**config)
            
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

class TensorLogger:
    """Specialized logger for tracking tensor shapes and statistics."""
    
    def __init__(self, parent_logger: logging.Logger):
        self.logger = parent_logger
        self._shape_logs = []
        self._lock = threading.Lock()
        
    def log_tensor(self, tensor: 'torch.Tensor', path: str, step: str) -> None:
        """Log tensor shape and statistics."""
        with self._lock:
            try:
                shape_info = {
                    'step': step,
                    'path': path,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device),
                    'stats': self._compute_tensor_stats(tensor)
                }
                self._shape_logs.append(shape_info)
            except Exception as e:
                self.logger.warning(f"Failed to log tensor: {str(e)}", exc_info=True)
    
    def log_dict(self, data: Dict, path: str = "", step: str = "") -> None:
        """Recursively log dictionary of tensors."""
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                self.log_dict(value, current_path, step)
            elif hasattr(value, 'shape'):  # Tensor-like object
                self.log_tensor(value, current_path, step)
    
    def _compute_tensor_stats(self, tensor: 'torch.Tensor') -> Dict[str, float]:
        """Compute basic statistics for a tensor."""
        try:
            with torch.no_grad():
                tensor_cpu = tensor.detach().float().cpu()
                return {
                    'min': float(tensor_cpu.min().item()),
                    'max': float(tensor_cpu.max().item()),
                    'mean': float(tensor_cpu.mean().item()),
                    'std': float(tensor_cpu.std().item()) if tensor_cpu.numel() > 1 else 0.0,
                    'numel': tensor_cpu.numel()
                }
        except Exception:
            return {'error': 'Failed to compute statistics'}
    
    def get_shape_history(self) -> List[Dict[str, Any]]:
        """Get the complete shape history."""
        with self._lock:
            return self._shape_logs.copy()
    
    def clear_logs(self) -> None:
        """Clear the shape history."""
        with self._lock:
            self._shape_logs.clear()

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output with detailed context."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.RED + Style.BRIGHT + Style.DIM
    }

    # Extended context fields for detailed logging
    CONTEXT_FIELDS = [
        'function', 'line_number', 'file_path',
        'input_data', 'expected_type', 'actual_type',
        'shape', 'device', 'dtype', 'error_type',
        'stack_info'
    ]
    
    # Fields to exclude from console output
    exclude_fields = ['process', 'thread', 'processName', 'threadName']
    
    def format(self, record):
        """Enhanced formatter with detailed context and error tracking."""
        # Create a copy of the record to avoid modifying the original
        filtered_record = logging.makeLogRecord(record.__dict__)
        
        # Get timestamp once for consistency
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Filter out excluded fields from the copy
        for field in self.exclude_fields:
            if hasattr(filtered_record, field):
                delattr(filtered_record, field)
        
        # Track error levels that cause failures
        if record.levelno >= logging.ERROR:
            with threading.Lock():  # Add thread safety
                _action_history[f'error_{time.time()}'] = {
                    'level': record.levelno,
                    'level_name': record.levelname,
                    'message': record.msg,
                    'timestamp': timestamp
                }
        
        # Build detailed context string
        context_parts = []
        for field in self.CONTEXT_FIELDS:
            if hasattr(record, field):
                value = getattr(record, field)
                if value is not None:
                    if isinstance(value, (dict, list, tuple)):
                        value_str = f"\n    {repr(value)}"
                    else:
                        value_str = str(value)
                    context_parts.append(f"{field}: {value_str}")
        
        # Add exception info and stack trace only when they exist
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            record.msg = f"{record.msg}\nException:\n{exception_text}"
        
        if record.levelno >= logging.ERROR and record.stack_info:
            stack_info_text = self.formatStack(record.stack_info)
            record.msg = f"{record.msg}\nStack Trace:\n{stack_info_text}"

        # Add context information if there are any context parts
        if context_parts:
            record.msg = f"{record.msg}\nContext:\n{chr(10).join(context_parts)}"
        
        # Get the base color for the entire message based on level
        base_color = self.COLORS.get(record.levelname, '')
        
        # Format the record using the parent formatter
        formatted_message = super().format(record)
        
        # Apply the color to the entire formatted message
        colored_message = f"{base_color}{formatted_message}{Style.RESET_ALL}"
        
        return colored_message
    
    KEYWORDS = {
        'start': (Fore.CYAN, ['Starting', 'Initializing', 'Beginning']),
        'success': (Fore.GREEN, ['Complete', 'Finished', 'Saved', 'Success']),
        'error': (Fore.RED, ['Error', 'Failed', 'Exception']),
        'warning': (Fore.YELLOW, ['Warning', 'Caution']),
        'progress': (Fore.BLUE, ['Processing', 'Loading', 'Computing'])
    }
    
    

def setup_logging(
    config: Optional[Union[LoggingConfig, Dict]] = None,
    log_dir: Optional[str] = None,
    level: Optional[Union[int, str]] = None,
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: Optional[bool] = None,
    propagate: Optional[bool] = None,
    console_level: Optional[Union[int, str]] = "INFO"
) -> Tuple[logging.Logger, TensorLogger]:
    """Setup logging configuration with detailed action tracking and colored output.
    
    Args:
        log_dir: Directory for log files
        level: Logging level for file output
        filename: Optional log file name
        module_name: Optional module name for logger
        capture_warnings: Whether to capture Python warnings
        propagate: Whether to propagate logs to parent loggers
        console_level: Logging level for console output (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create log directory only for file logging
    log_path = Path(log_dir)
    try:
        if filename:
            log_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created log directory: {log_path}")
    except Exception as e:
        print(f"{Fore.RED}Failed to create log directory: {str(e)}{Style.RESET_ALL}")
        raise
    
    # Create and configure logger
    logger_name = module_name or 'root'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Force DEBUG level for logger
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    try:
        # Create console handler with specified level
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',  # Remove exc_info from format
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(console_level)  # Use the console_level parameter
        logger.addHandler(console_handler)

        # Enable warning capture and configure propagation
        logging.captureWarnings(capture_warnings)
        logger.propagate = propagate
        
        # Add detailed file handler if filename provided
        if filename:
            file_path = log_path / filename
            file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | '
                '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
                '%(funcName)s |\n%(message)s\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            file_handler.setLevel(logging.DEBUG)  # Always capture full detail in file
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {file_path}", extra={
                'file_path': str(file_path),
                'log_level': logging.getLevelName(level)
            })
    except Exception as e:
        print(f"{Fore.RED}Failed to setup logging handlers: {str(e)}{Style.RESET_ALL}")
        raise
    
    # Suppress noisy loggers
    for logger_name in ["PIL", "torch", "transformers"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Log initialization only once
    if not logger.handlers:
        logger.info(f"Logging system initialized for {logger_name} at level {logging.getLevelName(level)}")
    
    # Create tensor logger
    tensor_logger = TensorLogger(logger)
        
    # Enable warning capture and configure propagation
    logging.captureWarnings(capture_warnings)
    logger.propagate = propagate
        
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
