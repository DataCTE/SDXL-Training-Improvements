from setuptools import setup, find_packages

setup(
    name="SDXL-Training-Improvements",
    version="0.1.0",
    description="A research-focused SDXL training framework implementing cutting-edge advances in diffusion model training",
    author="Alexander Rafael Izquierdo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0",
        "wandb>=0.15.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "pyyaml>=6.0.1",
        "bitsandbytes>=0.41.0",
        "xformers>=0.0.20",
        "einops>=0.6.1",
        "safetensors>=0.3.1",
        "scipy>=1.10.1",
        "matplotlib>=3.7.1",
        "torchvision>=0.15.1",
        "omegaconf>=2.3.0",
        "gradio>=3.39.0",
        "adamw-bf16"

    ]
)
