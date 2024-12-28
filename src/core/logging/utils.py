"""Enhanced logging utilities with detailed error tracking and formatting."""
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import colorama
from colorama import Fore, Style
from datetime import datetime

# Initialize colorama for Windows support
colorama.init(autoreset=True)

# Global action history dict with thread safety
_action_history: Dict[str, Any] = {}
_history_lock = threading.Lock()

class EnhancedFormatter(logging.Formatter):
    """Custom formatter with colored output and context tracking."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.RED + Style.BRIGHT + Style.DIM
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and context."""
        # Add timestamp
        record.created_fmt = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Get base color
        color = self.COLORS.get(record.levelname, '')
        
        # Format basic message
        message = super().format(record)
        
        # Add context information if available
        if hasattr(record, 'context'):
            context_str = '\n'.join(f"  {k}: {v}" for k, v in record.context.items())
            message = f"{message}\nContext:\n{context_str}"
            
        # Add exception info with proper formatting
        if record.exc_info:
            if not message.endswith('\n'):
                message += '\n'
            message += self.formatException(record.exc_info)
            
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and getattr(record, 'stack_info', None):
            if not message.endswith('\n'):
                message += '\n'
            message += f"Stack trace:\n{record.stack_info}"
            
        # Apply color
        return f"{color}{message}{Style.RESET_ALL}"

def create_enhanced_logger(
    name: str,
    level: str = "INFO",
    console_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    file_level: Optional[str] = None,
    capture_warnings: bool = True,
    track_actions: bool = True
) -> logging.Logger:
    """Create an enhanced logger with console and file output.
    
    Args:
        name: Logger name
        level: Base logging level
        console_level: Console handler level (defaults to level)
        log_file: Optional log file path
        file_level: File handler level (defaults to level)
        capture_warnings: Whether to capture Python warnings
        track_actions: Whether to track actions in history
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, (console_level or level).upper()))
    console_handler.setFormatter(EnhancedFormatter(
        '%(created_fmt)s | %(levelname)s | %(name)s | %(message)s'
    ))
    logger.addHandler(console_handler)
    
    # File handler if path provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, (file_level or level).upper()))
        file_handler.setFormatter(logging.Formatter(
            '%(created_fmt)s.%(msecs)03d | %(levelname)s | %(name)s | '
            '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
            '%(funcName)s |\n%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    # Configure warning capture
    if capture_warnings:
        logging.captureWarnings(True)
        
    def track_action(action: str, context: Optional[Dict] = None) -> None:
        """Track action in history with thread safety."""
        if track_actions:
            with _history_lock:
                timestamp = datetime.now().isoformat()
                _action_history[f"{timestamp}_{threading.get_ident()}"] = {
                    'action': action,
                    'context': context or {},
                    'logger': name,
                    'thread': threading.current_thread().name
                }
    
    # Add action tracking method
    logger.track_action = track_action
    
    return logger

def get_action_history() -> Dict[str, Any]:
    """Get copy of action history with thread safety."""
    with _history_lock:
        return _action_history.copy()

def clear_action_history() -> None:
    """Clear action history with thread safety."""
    with _history_lock:
        _action_history.clear()
