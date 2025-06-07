#!/usr/bin/env python3
"""
Simple inference script for SDXL models
"""

import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np

def load_pipeline(checkpoint_path, device="cuda", dtype=torch.float16):
    """Load SDXL pipeline with custom checkpoint."""
    
    # Base model path
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Load pipeline
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            unet=checkpoint_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16"
        )
    else:
        print("Loading base SDXL model")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16"
        )
    
    # Optimize pipeline
    pipe = pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    
    # Use DPM solver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    return pipe

def generate_images(
    pipe,
    prompt,
    negative_prompt="",
    num_images=1,
    num_inference_steps=25,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    seed=None
):
    """Generate images using the pipeline."""
    
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    ).images
    
    return images

def main():
    parser = argparse.ArgumentParser(description="SDXL Inference Script")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = load_pipeline(args.checkpoint, device, dtype)
    
    # Generate images
    print(f"Generating {args.num_images} image(s)...")
    images = generate_images(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_images=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    # Save images
    for i, image in enumerate(images):
        filename = f"image_{i:04d}.png"
        if args.seed is not None:
            filename = f"seed_{args.seed}_{filename}"
        
        output_path = output_dir / filename
        image.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"\nGeneration complete! Images saved to {output_dir}")

if __name__ == "__main__":
    main()