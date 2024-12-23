"""Training throughput monitoring utilities with extreme speedups."""
import time
import logging
from typing import Dict, Optional, Any, Union
from collections import deque

import torch

# Force speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)

class ThroughputMonitor:
    def __init__(
        self,
        window_size: int = 100,
        legacy_mode: bool = False,
        metric_prefix: str = "throughput/"
    ):
        self.window_size = window_size
        self.legacy_mode = legacy_mode
        self.metric_prefix = metric_prefix
        self.batch_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.last_time = time.time()
        self._total_samples = 0
        self._last_metrics: Dict[str, float] = {}

    def update(self, batch_size: Optional[Union[int, torch.Tensor]] = None, **kwargs: Any) -> None:
        current_time = time.time()
        if batch_size is not None:
            if isinstance(batch_size, torch.Tensor):
                batch_size = batch_size.item()
            self.batch_sizes.append(batch_size)
            self._total_samples += batch_size
        elif not self.legacy_mode:
            logger.warning("batch_size required when legacy_mode=False")
            return
        self.batch_times.append(current_time - self.last_time)
        self.last_time = current_time

    def get_metrics(self, include_legacy: bool = True) -> Dict[str, float]:
        metrics = {}
        if self.batch_times:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            total_samples = sum(self.batch_sizes) if self.batch_sizes else self._total_samples
            samples_per_sec = total_samples / sum(self.batch_times) if self.batch_sizes else (
                self._total_samples / sum(self.batch_times)
            )
            base_metrics = {
                "samples_per_sec": samples_per_sec,
                "batch_time_ms": avg_time * 1000,
                "accumulated_samples": total_samples
            }
            metrics.update({f"{self.metric_prefix}{k}": v for k, v in base_metrics.items()})
            if include_legacy and self.legacy_mode:
                metrics.update(base_metrics)
            self._last_metrics = metrics
        return metrics if metrics else self._last_metrics

# Compile for extreme speedups if available
if hasattr(torch, "compile"):
    ThroughputMonitor.update = torch.compile(
        ThroughputMonitor.update, mode="reduce-overhead", fullgraph=True
    )
    ThroughputMonitor.get_metrics = torch.compile(
        ThroughputMonitor.get_metrics, mode="reduce-overhead", fullgraph=True
    )
