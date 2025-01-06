"""Progress tracking utilities for logging."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tqdm import tqdm

@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    total: int
    desc: str = ""
    smoothing: float = 0.3
    disable: bool = False
    position: int = 0
    leave: bool = True

class ProgressTracker:
    """Tracks progress with metrics."""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        """Initialize progress tracker."""
        if config is None:
            config = ProgressConfig(total=0)
            
        self.progress = tqdm(
            total=config.total,
            desc=config.desc,
            smoothing=config.smoothing,
            disable=config.disable,
            position=config.position,
            leave=config.leave
        )
        
    def update(self, n: int = 1, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update progress with optional metrics."""
        self.progress.update(n)
        if metrics:
            self.progress.set_postfix(metrics, refresh=True)
            
    def close(self) -> None:
        """Close progress bar."""
        self.progress.close()
