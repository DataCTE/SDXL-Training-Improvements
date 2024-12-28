"""Centralized metrics collection and logging for SDXL training."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import torch
import torch.distributed as dist
from collections import defaultdict
import time
import numpy as np
from functools import wraps
from .base import get_logger, LogConfig, reduce_dict
from .wandb import WandbLogger

logger = get_logger(__name__)

def make_picklable(func):
    """Decorator to make functions picklable."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

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
    
    def __init__(
        self, 
        window_size: int = 100,
        wandb_logger: Optional[WandbLogger] = None,
        log_prefix: str = ""
    ):
        self.window_size = window_size
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
        self.step_times = []
        self.current_metrics = TrainingMetrics()
        self.wandb_logger = wandb_logger
        self.log_prefix = log_prefix
        
    def update(self, metrics: Dict[str, Any], batch_size: Optional[int] = None) -> None:
        """Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
            batch_size: Optional batch size for throughput calculation
        """
        try:
            # Update timing
            current_time = time.time()
            if self.step_times:
                step_time = current_time - self.step_times[-1]
                self.step_times.append(current_time)
                if len(self.step_times) > self.window_size:
                    self.step_times.pop(0)
            else:
                self.step_times.append(current_time)
            
            # Update metrics history
            for k, v in metrics.items():
                try:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    elif not isinstance(v, (int, float)):
                        continue
                        
                    self.metrics_history[k].append(float(v))
                    if len(self.metrics_history[k]) > self.window_size:
                        self.metrics_history[k].pop(0)
                except Exception as e:
                    logger.warning(f"Failed to process metric {k}: {str(e)}")
            
            # Update sample count
            if batch_size is not None:
                self.total_samples += batch_size
                
        except Exception as e:
            logger.error(
                "Failed to update metrics",
                exc_info=True,
                extra={
                    'metrics': metrics,
                    'batch_size': batch_size,
                    'error': str(e)
                }
            )
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics with moving averages."""
        try:
            metrics = {}
            
            # Compute basic statistics
            for k, v in self.metrics_history.items():
                if v:  # Only process non-empty histories
                    metrics[k] = float(np.mean(v))
                    metrics[f"{k}_std"] = float(np.std(v))
                    
            # Add throughput metrics if available
            if self.step_times and len(self.step_times) > 1:
                steps_per_second = 1.0 / np.mean(np.diff(self.step_times))
                metrics.update({
                    "steps_per_second": steps_per_second,
                    "samples_per_second": steps_per_second * self.total_samples / len(self.step_times),
                    "step_time_ms": 1000.0 / steps_per_second
                })
                
            # Add memory metrics if available
            if torch.cuda.is_available():
                metrics.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
                })
                
            return metrics
            
        except Exception as e:
            logger.error(
                "Failed to compute metrics",
                exc_info=True,
                extra={'error': str(e)}
            )
            return {}

@make_picklable
def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    is_main_process: bool = True,
    use_wandb: bool = False,
    wandb_logger: Optional[WandbLogger] = None,
    step_type: str = "step",
    additional_metrics: Optional[Dict[str, Any]] = None,
    metric_prefix: Optional[str] = None
) -> None:
    """Log training metrics to console and optional trackers.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        is_main_process: Whether this is the main training process
        use_wandb: Whether to log to Weights & Biases
        wandb_logger: Optional WandbLogger instance
        step_type: Type of step (step/epoch)
        additional_metrics: Optional additional metrics to log
        metric_prefix: Optional prefix for metric names
    """
    try:
        # Reduce metrics across processes if distributed
        if dist.is_initialized():
            metrics = reduce_dict(metrics)
            
        if not is_main_process:
            return
            
        # Add prefix if specified
        if metric_prefix:
            metrics = {f"{metric_prefix}/{k}": v for k, v in metrics.items()}
            
        # Add additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
            
        # Format metrics for logging
        log_items = []
        for k, v in sorted(metrics.items()):
            try:
                if isinstance(v, (int, bool)):
                    log_items.append(f"{k}: {v}")
                elif isinstance(v, float):
                    log_items.append(f"{k}: {v:.4f}")
                elif isinstance(v, torch.Tensor):
                    log_items.append(f"{k}: {v.item():.4f}")
                else:
                    log_items.append(f"{k}: {v}")
            except Exception as e:
                logger.warning(f"Failed to format metric {k}: {str(e)}")
                
        # Log to console
        metric_str = f"{step_type.capitalize()} {step} | " + " | ".join(log_items)
        logger.info(metric_str)
        
        # Log to wandb if enabled
        if use_wandb and wandb_logger is not None:
            try:
                wandb_logger.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(
                    "Failed to log metrics to wandb",
                    exc_info=True,
                    extra={
                        'step': step,
                        'error': str(e)
                    }
                )
                
    except Exception as e:
        logger.error(
            "Failed to log metrics",
            exc_info=True,
            extra={
                'step': step,
                'error': str(e),
                'metrics': metrics
            }
        )
