"""Progress tracking utilities for logging."""
import time
from collections import deque
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import numpy as np
from .base import ProgressConfig

class ProgressTracker:
    """Tracks progress with throughput monitoring."""
    
    def __init__(self, config: ProgressConfig):
        """Initialize progress tracker."""
        self.config = config
        self.progress = tqdm(
            total=config.total,
            desc=config.desc,
            smoothing=config.smoothing,
            disable=config.disable,
            position=config.position,
            leave=config.leave
        )
        self.window_size = config.window_size
        self.steps = deque(maxlen=config.window_size)
        self.batch_sizes = deque(maxlen=config.window_size)
        self.last_update = time.time()
        self.metric_prefix = config.metric_prefix
        self._total_samples = 0
        self._last_metrics = {}
        
    def update(self, n: int = 1, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Update progress and return metrics."""
        self.progress.update(n)
        current = time.time()
        
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
            self._total_samples += batch_size
            
        elapsed = current - self.last_update
        self.steps.append(elapsed / n)
        self.last_update = current
        
        metrics = self.get_metrics()
        self.progress.set_postfix(metrics, refresh=True)
        return metrics
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current throughput metrics."""
        if not self.steps:
            return self._last_metrics
            
        avg_time = sum(self.steps) / len(self.steps)
        total_samples = sum(self.batch_sizes) if self.batch_sizes else self._total_samples
        samples_per_sec = (
            total_samples / sum(self.steps) if self.batch_sizes 
            else self._total_samples / sum(self.steps)
        )
        
        metrics = {
            f"{self.metric_prefix}samples_per_sec": samples_per_sec,
            f"{self.metric_prefix}batch_time_ms": avg_time * 1000,
            f"{self.metric_prefix}accumulated_samples": total_samples
        }
        self._last_metrics = metrics
        return metrics
        
    def close(self) -> None:
        """Clean up progress bar."""
        self.progress.close()
