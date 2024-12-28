"""Logging configuration for training with detailed action tracking."""
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
import colorama
from colorama import Fore, Style
from datetime import datetime
from pathlib import Path
import threading
# Initialize colorama for Windows support
colorama.init(autoreset=True)


# Track actions and their timing
action_history: Dict[str, Any] = {}

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output with detailed context."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.RED + Style.BRIGHT + Style.DIM
    }

    def format(self, record):
        """Enhanced formatter with detailed context and error tracking."""
        # Always show DEBUG messages with visual separator
        if record.levelno == logging.DEBUG:
            record.msg = f"[DEBUG] {record.msg}"

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
                action_history[f'error_{time.time()}'] = {
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
    log_dir: str = "outputs/wslref/logs",
    level: int = logging.DEBUG,  # Keep DEBUG as default for file logging
    filename: Optional[str] = None,
    module_name: Optional[str] = None,
    capture_warnings: bool = True,
    propagate: bool = True,
    console_level: int = logging.INFO  # Add new parameter for console output level
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
        'actions': action_history,
        'total_actions': len(action_history),
        'categories': {
            category: len([a for a in action_history.values() if a['category'] == category])
            for category in set(a['category'] for a in action_history.values())
        }
    }
