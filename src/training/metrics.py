"""Training metrics logging and tracking utilities."""
import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from ..core.distributed import reduce_dict
from ..core.logging.wandb import WandbLogger

logger = logging.getLogger(__name__)

def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    is_main_process: bool = True,
    use_wandb: bool = False,
    wandb_logger: Optional[WandbLogger] = None,
    step_type: str = "step"
) -> None:
    """Log training metrics to console and optional trackers.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        is_main_process: Whether this is the main training process
        use_wandb: Whether to log to Weights & Biases
        wandb_logger: Optional WandbLogger instance
        step_type: Type of step (step/epoch)
    """
    try:
        # Reduce metrics across processes
        if dist.is_initialized():
            metrics = reduce_dict(metrics)
            
        if not is_main_process:
            return
            
        # Log to console
        metric_str = f"{step_type.capitalize()} {step}: "
        metric_str += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                              for k, v in metrics.items())
        logger.info(metric_str)
        
        # Log to wandb if enabled
        if use_wandb and wandb_logger is not None:
            try:
                wandb_logger.log_metrics(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")
