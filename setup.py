from setuptools import setup, find_packages

setup(
    name="sdxl_trainer",
    version="0.1.0",
    description="High-performance SDXL training framework optimized for WSL",
    author="Your Name",
    author_email="your.email@example.com",
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
        "colorama>=0.4.6",
        "packaging>=23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
