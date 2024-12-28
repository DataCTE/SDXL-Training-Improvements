"""Validation utilities for SDXL training."""
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from src.core.logging import get_logger, LogConfig
from src.models.sdxl import StableDiffusionXLModel

logger = get_logger(__name__)

class ValidationLogger:
    """Handles validation metrics and image generation."""
    
    def __init__(
        self,
        model: StableDiffusionXLModel,
        prompts: List[str],
        output_dir: Path,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ):
        self.model = model
        self.prompts = prompts
        self.output_dir = output_dir
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Create validation directory
        self.validation_dir = output_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def run_validation(
        self,
        step: int,
        wandb_logger: Optional[Any] = None
    ) -> Dict[str, float]:
        """Run validation and log results."""
        try:
            logger.info(f"Running validation at step {step}")
            
            self.model.eval()
            metrics = {}
            images = []
            
            for idx, prompt in enumerate(self.prompts):
                try:
                    # Generate image
                    image = self.model(
                        prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale
                    ).images[0]
                    
                    # Save image
                    image_path = self.validation_dir / f"step_{step}_sample_{idx}.png"
                    image.save(image_path)
                    images.append(image)
                    
                    logger.debug(f"Generated validation image {idx} at {image_path}")
                    
                except Exception as e:
                    logger.error(
                        f"Failed to generate validation image {idx}",
                        exc_info=True,
                        extra={
                            'prompt': prompt,
                            'error': str(e)
                        }
                    )
            
            # Log to wandb if available
            if wandb_logger is not None and images:
                wandb_logger.log({
                    "validation/images": [wandb_logger.Image(img) for img in images],
                    "validation/step": step
                })
            
            self.model.train()
            return metrics
            
        except Exception as e:
            logger.error(
                "Validation failed",
                exc_info=True,
                extra={
                    'step': step,
                    'error': str(e)
                }
            )
            return {}
