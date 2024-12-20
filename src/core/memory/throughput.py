"""Training throughput monitoring utilities."""
import time
from typing import Dict, Optional
import torch
from collections import deque

class ThroughputMonitor:
    """Monitors training throughput metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.batch_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self, batch_size: int) -> None:
        """Update metrics with new batch."""
        current_time = time.time()
        self.batch_times.append(current_time - self.last_time)
        self.batch_sizes.append(batch_size)
        self.last_time = current_time
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current throughput metrics."""
        if not self.batch_times:
            return {}
            
        # Calculate metrics
        avg_time = sum(self.batch_times) / len(self.batch_times)
        total_samples = sum(self.batch_sizes)
        samples_per_sec = total_samples / sum(self.batch_times)
        
        return {
            "throughput/samples_per_sec": samples_per_sec,
            "throughput/batch_time_ms": avg_time * 1000,
            "throughput/accumulated_samples": total_samples
        }
