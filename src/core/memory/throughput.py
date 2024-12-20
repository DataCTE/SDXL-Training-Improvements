"""Training throughput monitoring utilities."""
import time
import logging
from typing import Dict, Optional, Any, Union
import torch
from collections import deque

logger = logging.getLogger(__name__)

class ThroughputMonitor:
    """Monitors training throughput metrics with backwards compatibility."""
    
    def __init__(
        self,
        window_size: int = 100,
        legacy_mode: bool = False,
        metric_prefix: str = "throughput/"
    ):
        """Initialize throughput monitor.
        
        Args:
            window_size: Size of sliding window for metrics
            legacy_mode: Enable compatibility with older integrations
            metric_prefix: Prefix for metric names
        """
        self.window_size = window_size
        self.legacy_mode = legacy_mode
        self.metric_prefix = metric_prefix
        
        # Metric tracking
        self.batch_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.last_time = time.time()
        
        # Legacy state
        self._total_samples = 0
        self._last_metrics: Dict[str, float] = {}
        
    def update(
        self,
        batch_size: Optional[Union[int, torch.Tensor]] = None,
        **kwargs: Any
    ) -> None:
        """Update metrics with new batch.
        
        Args:
            batch_size: Size of current batch (optional in legacy mode)
            **kwargs: Additional metrics for backwards compatibility
        """
        try:
            current_time = time.time()
            
            # Handle batch size
            if batch_size is not None:
                if isinstance(batch_size, torch.Tensor):
                    batch_size = batch_size.item()
                self.batch_sizes.append(batch_size)
                self._total_samples += batch_size
            elif not self.legacy_mode:
                logger.warning("batch_size required when legacy_mode=False")
                return
                
            # Update timing
            self.batch_times.append(current_time - self.last_time)
            self.last_time = current_time
            
        except Exception as e:
            logger.warning(f"Error updating throughput metrics: {str(e)}")
            
    def get_metrics(self, include_legacy: bool = True) -> Dict[str, float]:
        """Get current throughput metrics.
        
        Args:
            include_legacy: Include legacy format metrics
            
        Returns:
            Dict of current metrics
        """
        metrics = {}
        
        try:
            if self.batch_times:
                # Calculate metrics
                avg_time = sum(self.batch_times) / len(self.batch_times)
                total_samples = sum(self.batch_sizes) if self.batch_sizes else self._total_samples
                samples_per_sec = (
                    total_samples / sum(self.batch_times)
                    if self.batch_sizes else
                    self._total_samples / sum(self.batch_times)
                )
                
                # Store metrics
                base_metrics = {
                    "samples_per_sec": samples_per_sec,
                    "batch_time_ms": avg_time * 1000,
                    "accumulated_samples": total_samples
                }
                
                # Add prefixed metrics
                metrics.update({
                    f"{self.metric_prefix}{k}": v 
                    for k, v in base_metrics.items()
                })
                
                # Add legacy format if requested
                if include_legacy and self.legacy_mode:
                    metrics.update(base_metrics)
                    
                self._last_metrics = metrics
                
            return metrics if metrics else self._last_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {str(e)}")
            return self._last_metrics
