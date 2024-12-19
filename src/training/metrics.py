"""Training metrics logging and tracking utilities."""
import logging
from typing import Any, Dict, Optional

import wandb

logger = logging.getLogger(__name__)

def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    is_main_process: bool = True,
    use_wandb: bool = False,
    step_type: str = "step"
) -> None:
    """Log training metrics to console and optional trackers.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        is_main_process: Whether this is the main training process
        use_wandb: Whether to log to Weights & Biases
        step_type: Type of step (step/epoch)
    """
    try:
        if not is_main_process:
            return
            
        # Log to console
        metric_str = f"{step_type.capitalize()} {step}: "
        metric_str += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                              for k, v in metrics.items())
        logger.info(metric_str)
        
        # Log to wandb if enabled
        if use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to wandb: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")
