"""Base logging configuration."""
from dataclasses import dataclass, field
from typing import List, Optional
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
    
    # Metrics settings
    metrics_window: int = 100
    metrics_history: bool = True

    def copy(self) -> 'LogConfig':
        """Create a copy of the config."""
        return LogConfig(**self.__dict__)
