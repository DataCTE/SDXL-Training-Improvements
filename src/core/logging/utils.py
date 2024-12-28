"""Basic logging utilities to avoid circular imports."""
import logging
import sys
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows support
colorama.init(autoreset=True)

def create_basic_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a basic logger without file handling or complex config."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
        
    return logger
