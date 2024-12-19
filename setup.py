from setuptools import setup, find_packages

setup(
    name="sdxl_trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "nvidia-dali-cuda110>=1.24.0",
    ],
    python_requires=">=3.8",
)
