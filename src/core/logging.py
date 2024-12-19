"""Logging configuration for training."""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    filename: Optional[str] = None
) -> None:
    """Setup logging configuration."""
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if filename:
        handlers.append(
            logging.FileHandler(
                log_path / filename,
                mode='a'
            )
        )
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress some common noisy loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
def cleanup_logging() -> None:
    """Cleanup logging handlers."""
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
