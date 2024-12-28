"""Logging configuration dataclass."""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass 
class LogConfig:
    """Unified logging configuration."""
    # Basic logging config
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "outputs/logs"
    filename: Optional[str] = "training.log"
    capture_warnings: bool = True
    console_output: bool = True
    file_output: bool = True
    log_cuda_memory: bool = True
    log_system_memory: bool = True
    performance_logging: bool = True
    propagate: bool = True
    
    # Metrics config
    metrics_window_size: int = 100
    
    # W&B config
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    
    @classmethod
    def from_global_config(cls, config) -> 'LogConfig':
        """Create LogConfig from GlobalConfig.LoggingConfig."""
        return cls(
            console_level=config.console_level,
            file_level=config.file_level,
            log_dir=config.log_dir,
            filename=config.filename,
            capture_warnings=config.capture_warnings,
            console_output=config.console_output,
            file_output=config.file_output,
            log_cuda_memory=config.log_cuda_memory,
            log_system_memory=config.log_system_memory,
            performance_logging=config.performance_logging,
            propagate=config.propagate,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name,
            wandb_tags=config.wandb_tags,
            wandb_notes=config.wandb_notes
        )
