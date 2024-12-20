"""Logging configuration for training with detailed action tracking."""
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import colorama
from colorama import Fore, Style
from datetime import datetime

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
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    KEYWORDS = {
        'start': (Fore.CYAN, ['Starting', 'Initializing', 'Beginning']),
        'success': (Fore.GREEN, ['Complete', 'Finished', 'Saved', 'Success']),
        'error': (Fore.RED, ['Error', 'Failed', 'Exception']),
        'warning': (Fore.YELLOW, ['Warning', 'Caution']),
        'progress': (Fore.BLUE, ['Processing', 'Loading', 'Computing'])
    }
    
    def format(self, record):
        # Add timestamp and context
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        record.msg = f"[{timestamp}] {record.msg}"
        
        # Add color to level name
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            record.levelname = f"{color}{record.levelname:8}{Style.RESET_ALL}"
        
        # Add color to important keywords and track actions
        msg = record.msg
        for category, (color, keywords) in self.KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in msg.lower():
                    # Track action timing
                    action_key = f"{category}_{time.time()}"
                    action_history[action_key] = {
                        'category': category,
                        'message': msg,
                        'timestamp': timestamp
                    }
                    # Add color
                    msg = msg.replace(keyword, f"{color}{keyword}{Style.RESET_ALL}")
        record.msg = msg
        
        # Add context information for errors
        if record.levelno >= logging.ERROR:
            if hasattr(record, 'exc_info') and record.exc_info:
                record.msg = f"{record.msg}\nException details: {str(record.exc_info[1])}"
            
        return super().format(record)

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    filename: Optional[str] = None,
    module_name: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration with detailed action tracking and colored output.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        filename: Optional log file name
        module_name: Optional module name for logger
        
    Returns:
        Configured logger instance
    """
    # Create log directory only for file logging
    log_path = Path(log_dir)
    if filename:
        log_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created log directory: {log_path}")
    except Exception as e:
        print(f"{Fore.RED}Failed to create log directory: {str(e)}{Style.RESET_ALL}")
        raise
    
    # Create and configure logger
    logger_name = module_name or 'root'
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    try:
        # Create console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(
            '%(levelname)s | %(name)s | %(message)s'
        ))
        logger.addHandler(console_handler)
        
        # Add file handler if filename provided
        if filename:
            file_path = log_path / filename
            file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            ))
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {file_path}")
    except Exception as e:
        print(f"{Fore.RED}Failed to setup logging handlers: {str(e)}{Style.RESET_ALL}")
        raise
    
    # Suppress noisy loggers
    for logger_name in ["PIL", "torch", "transformers"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Log initialization only once
    if not logger.handlers:
        logger.info(f"Logging system initialized for {logger_name} at level {logging.getLevelName(level)}")
    
    return logger
    
def cleanup_logging() -> Dict[str, Any]:
    """Cleanup logging handlers and return action history.
    
    Returns:
        Dictionary containing logged action history
    """
    # Close all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except Exception as e:
            print(f"{Fore.RED}Error closing handler: {str(e)}{Style.RESET_ALL}")
    
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
