"""Logging configuration for training."""
import logging
import sys
from pathlib import Path
from typing import Optional
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows support
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
            
        # Add color to important keywords in message
        if "Starting" in record.msg:
            record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
        elif "Complete" in record.msg or "Finished" in record.msg or "Saved" in record.msg:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif "Error" in record.msg:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
            
        return super().format(record)

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    filename: Optional[str] = None
) -> None:
    """Setup logging configuration with colored console output."""
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    handlers = [console_handler]
    
    # Add file handler if filename provided
    if filename:
        file_handler = logging.FileHandler(log_path / filename, mode='a')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers and add new ones
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Suppress some common noisy loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    # Log initial setup message
    logging.info("Logging system initialized with colored console output")
    
def cleanup_logging() -> None:
    """Cleanup logging handlers."""
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
