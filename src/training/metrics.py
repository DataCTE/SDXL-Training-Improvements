"""Centralized metrics collection and logging for SDXL training."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from collections import defaultdict
import time
import numpy as np
from src.core.logging import setup_logging

logger = setup_logging(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    steps_per_second: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "steps_per_second": self.steps_per_second,
            "samples_per_second": self.samples_per_second,
            "gpu_memory_allocated_gb": self.gpu_memory_allocated / (1024**3),
            "gpu_memory_reserved_gb": self.gpu_memory_reserved / (1024**3)
        }

class MetricsLogger:
    """Centralized metrics logging with moving averages."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
        self.step_times = []
        self.current_metrics = TrainingMetrics()
        
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        try:
            # Update step timing
            current_time = time.time()
            if self.step_times:
                step_time = current_time - self.step_times[-1]
                self.step_times.append(current_time)
                if len(self.step_times) > self.window_size:
                    self.step_times.pop(0)
            else:
                self.step_times.append(current_time)
            
            # Update metrics
            self.current_metrics.loss = metrics.get("loss", 0.0)
            self.current_metrics.grad_norm = metrics.get("grad_norm", 0.0)
            self.current_metrics.learning_rate = metrics.get("learning_rate", 0.0)
            self.current_metrics.batch_size = metrics.get("batch_size", 0)
            
            # Calculate throughput
            if len(self.step_times) > 1:
                steps_per_second = 1.0 / np.mean(np.diff(self.step_times))
                self.current_metrics.steps_per_second = steps_per_second
                self.current_metrics.samples_per_second = steps_per_second * self.current_metrics.batch_size
            
            # Update GPU memory metrics if available
            if torch.cuda.is_available():
                self.current_metrics.gpu_memory_allocated = torch.cuda.memory_allocated()
                self.current_metrics.gpu_memory_reserved = torch.cuda.memory_reserved()
            
            # Store history
            for k, v in self.current_metrics.to_dict().items():
                self.metrics_history[k].append(v)
                if len(self.metrics_history[k]) > self.window_size:
                    self.metrics_history[k].pop(0)
                    
            logger.debug(f"Updated metrics: {self.get_current_metrics()}")
            
        except Exception as e:
            logger.error(
                "Failed to update metrics",
                exc_info=True,
                extra={
                    'metrics': metrics,
                    'error': str(e)
                }
            )
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics with moving averages."""
        current = {}
        try:
            for k, v in self.metrics_history.items():
                if v:
                    current[k] = np.mean(v)
                    current[f"{k}_std"] = np.std(v)
            return current
        except Exception as e:
            logger.error(
                "Failed to compute metrics",
                exc_info=True,
                extra={'error': str(e)}
            )
            return {}

    def log_metrics(
        self,
        step: int,
        wandb_logger: Optional[Any] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log current metrics."""
        try:
            metrics = self.get_current_metrics()
            if additional_metrics:
                metrics.update(additional_metrics)
            
            # Log to console
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {step} | {metrics_str}")
            
            # Log to wandb if available
            if wandb_logger is not None:
                wandb_logger.log(metrics, step=step)
                
        except Exception as e:
            logger.error(
                "Failed to log metrics",
                exc_info=True,
                extra={
                    'step': step,
                    'metrics': metrics if 'metrics' in locals() else None,
                    'error': str(e)
                }
            )
