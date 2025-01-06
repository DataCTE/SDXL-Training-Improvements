"""Base logging configuration."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class LogConfig:
    """Unified logging configuration."""
    # Core settings
    name: str = "root"
    log_dir: Path = Path("outputs/logs")
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    filename: str = "training.log"
    
    # Feature flags
    enable_console: bool = True
    enable_file: bool = True
    enable_wandb: bool = False
    enable_progress: bool = True
    enable_metrics: bool = True
    enable_memory: bool = True
    capture_warnings: bool = True
    propagate: bool = False
    
    # W&B settings
    wandb_project: str = "default"
    wandb_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Progress settings
    progress_window: int = 100
    progress_smoothing: float = 0.3
    progress_position: int = 0
    progress_leave: bool = True
    
    # Metrics settings
    metrics_window: int = 100
    metrics_history: bool = True
    metrics_prefix: str = "metrics/"

    def copy(self) -> 'LogConfig':
        """Create a copy of the config."""
        return LogConfig(**self.__dict__)

@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""
    window_size: int = 100
    keep_history: bool = True
    prefix: str = "metrics/"

@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    total: int
    desc: str = ""
    smoothing: float = 0.3
    disable: bool = False
    position: int = 0
    leave: bool = True
    window_size: int = 100
    metric_prefix: str = "throughput/"

class LoggingError(Exception):
    """Base exception for logging errors."""
    pass

class ConfigurationError(LoggingError):
    """Error in logging configuration."""
    pass

class MetricsError(LoggingError):
    """Error in metrics tracking."""
    pass
