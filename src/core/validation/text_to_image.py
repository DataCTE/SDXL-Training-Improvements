"""Image generation validation utilities for SDXL training."""
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
import matplotlib.pyplot as plt

from ...models import StableDiffusionXLModel, ModelType
from ...config import Config
from .types import ValidationMode, ValidationMetric, ValidationConfig

logger = logging.getLogger(__name__)

class TextToImageValidator:
    """Generates validation images during training."""
    
    def __init__(
        self,
        config: Config,
        device: Union[str, torch.device],
        output_dir: Union[str, Path],
        validation_prompts: Optional[list] = None
    ):
        """Initialize validator.
        
        Args:
            config: Training configuration
            device: Target device
            output_dir: Directory to save validation images
            validation_prompts: List of prompts to use for validation
        """
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir) / "validation_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default validation prompts if none provided
        self.validation_prompts = validation_prompts or [
            "a photo of a cat sitting on a windowsill at sunset",
            "an oil painting of a medieval castle on a hilltop",
            "a detailed pencil sketch of a blooming rose"
        ]
        
        # Load base model for comparison
        logger.info("Loading base model for validation...")
        self.base_model = StableDiffusionXLModel(ModelType.BASE)
        self.base_model.from_pretrained(
            config.model.pretrained_model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        self.base_model.to(device)
        
        # Cache base model outputs
        logger.info("Generating base model reference images...")
        self.base_images = self._generate_base_images()
        
    def _generate_base_images(self) -> dict:
        """Generate and cache base model outputs."""
        base_images = {}
        
        for prompt in self.validation_prompts:
            with torch.no_grad():
                image = self.base_model.generate(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=1024,
                    height=1024
                )
                base_images[prompt] = image
                
        return base_images
        
    def validate(
        self,
        model: StableDiffusionXLModel,
        step: int,
        seed: Optional[int] = None
    ) -> None:
        """Generate validation images and compare to base model.
        
        Args:
            model: Current training model
            step: Current training step
            seed: Optional seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Create output directory for this step
        step_dir = self.output_dir / f"step_{step}"
        step_dir.mkdir(exist_ok=True)
        
        for prompt in self.validation_prompts:
            # Generate image with current model
            with torch.no_grad():
                current_image = model.generate(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    width=1024,
                    height=1024
                )
                
            # Create comparison figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            ax1.imshow(self.base_images[prompt])
            ax1.set_title("Base Model")
            ax1.axis('off')
            
            ax2.imshow(current_image)
            ax2.set_title(f"Step {step}")
            ax2.axis('off')
            
            plt.suptitle(f'Prompt: "{prompt}"', fontsize=16)
            
            # Save comparison
            comparison_path = step_dir / f"{prompt[:50]}.png"
            plt.savefig(comparison_path)
            plt.close(fig)
            
            # Save individual images
            self.base_images[prompt].save(step_dir / f"{prompt[:50]}_base.png")
            current_image.save(step_dir / f"{prompt[:50]}_current.png")
            
        logger.info(f"Saved validation images to {step_dir}")
