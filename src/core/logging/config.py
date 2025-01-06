"""Logging configuration dataclass."""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

@dataclass
class LogConfig:
    """Unified logging configuration."""
    # Basic logging
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "outputs/logs"
    filename: Optional[str] = "training.log"
    capture_warnings: bool = True
    console_output: bool = True
    file_output: bool = True
    propagate: bool = True
    
    # Progress tracking
    progress_tracking: bool = True
    progress_history_aware: bool = False
    progress_history_path: Optional[str] = "outputs/logs/progress_history.json"
    progress_bottleneck_threshold: float = 1.5
    progress_smoothing: float = 0.3
    
    # Metrics
    metrics_window_size: int = 100
    log_cuda_memory: bool = True
    log_system_memory: bool = True
    performance_logging: bool = True
    
    # W&B integration
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    
    # Additional settings
    segment_names: List[str] = field(default_factory=lambda: ["main"])
    max_history: int = 1000
