"""Image generation validation utilities for SDXL training."""
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TextToImageValidator:
    """Generates validation images during training."""
    
    def __init__(
        self,
        base_model_path: str,
        device: Union[str, torch.device],
        output_dir: Union[str, Path],
        validation_prompts: Optional[list] = None
    ):
        """Initialize validator.
        
        Args:
            base_model_path: Path to base model for comparison
            device: Target device
            output_dir: Directory to save validation images
            validation_prompts: List of prompts to use for validation
        """
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
        self.base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            variant="fp16" if device.type == "cuda" else None
        ).to(device)
        
        # Cache base model outputs
        logger.info("Generating base model reference images...")
        self.base_images = self._generate_base_images()
        
    def _generate_base_images(self) -> dict:
        """Generate and cache base model outputs."""
        base_images = {}
        
        for prompt in self.validation_prompts:
            with torch.no_grad():
                image = self.base_pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
                base_images[prompt] = image
                
        return base_images
        
    def validate(
        self,
        pipeline: StableDiffusionXLPipeline,
        step: int,
        seed: Optional[int] = None
    ) -> None:
        """Generate validation images and compare to base model.
        
        Args:
            pipeline: Current training pipeline
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
                current_image = pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
                
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
            plt.close()
            
            # Save individual images
            self.base_images[prompt].save(step_dir / f"{prompt[:50]}_base.png")
            current_image.save(step_dir / f"{prompt[:50]}_current.png")
            
        logger.info(f"Saved validation images to {step_dir}")
