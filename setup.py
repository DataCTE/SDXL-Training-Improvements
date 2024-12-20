from setuptools import setup, find_packages

setup(
    name="sdxl-trainer",
    version="0.1.0",
    description="Research-focused SDXL training framework with novel methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander Izquierdo",
    author_email="izquierdoxanderl@gmail.com",
    url="https://github.com/DataCTE/SDXL-Training-Improvements",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.24.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "pyyaml>=6.0.1",
        "wandb>=0.16.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "nvidia-dali-cuda110>=1.24.0",
        "colorama>=0.4.6",
        "packaging>=23.2",
        "safetensors>=0.4.0",
        "datasets>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "ruff>=0.1.6",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sdxl-train=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", 
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, stable-diffusion, diffusion-models, pytorch",
)
