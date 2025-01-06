"""Metrics tracking utilities for logging."""
import threading
from collections import deque
from typing import Optional, Dict, List, Any
import numpy as np
from .base import MetricsConfig

class MetricsTracker:
    """Tracks metrics with configurable window size and history."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize metrics tracker."""
        self.config = config
        self.metrics: Dict[str, deque] = {}
        self.history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        
    def update(self, name: str, value: float) -> None:
        """Update metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.config.window_size)
                if self.config.keep_history:
                    self.history[name] = []
                    
            self.metrics[name].append(value)
            if self.config.keep_history:
                self.history[name].append(value)
                
    def get_average(self, name: str) -> Optional[float]:
        """Get current average for metric."""
        values = self.metrics.get(name)
        if values:
            return float(np.mean(values))
        return None
        
    def get_all_averages(self) -> Dict[str, float]:
        """Get current averages for all metrics."""
        return {
            name: float(np.mean(values))
            for name, values in self.metrics.items()
        }
        
    def get_history(self, name: str) -> Optional[List[float]]:
        """Get full history for metric if available."""
        return self.history.get(name)
        
    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()
            self.history.clear()