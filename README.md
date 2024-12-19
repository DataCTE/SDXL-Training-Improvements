# SDXL Training Framework

A high-performance training framework for Stable Diffusion XL models, optimized for WSL environments.

## Features

- Efficient memory management with layer offloading
- Distributed training support
- Advanced data preprocessing pipeline
- Tag-based loss weighting
- Flow matching and DDPM training methods
- Wandb integration for experiment tracking
- CUDA-optimized tensor operations

## Requirements

- WSL2 with Ubuntu 20.04 or later
- CUDA 11.7 or later
- Python 3.8+
- 24GB+ VRAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sdxl-training.git
cd sdxl-training
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

1. Prepare your configuration file:
```bash
cp src/config.yaml my_config.yaml
```

2. Edit the configuration file to match your requirements:
- Set data paths
- Configure training parameters
- Adjust memory optimization settings

3. Start training:
```bash
python src/main.py --config my_config.yaml
```

## Configuration

The framework uses a hierarchical YAML configuration system with the following main sections:

- `global_config`: General settings and paths
- `model`: Model architecture and parameters
- `training`: Training loop configuration
- `data`: Dataset and preprocessing settings
- `tag_weighting`: Caption-based loss weighting

See `src/config.yaml` for a complete example.

## Memory Optimization

The framework includes several memory optimization features:

- Layer offloading to CPU
- Gradient checkpointing
- Mixed precision training
- Activation offloading
- Efficient tensor management

Configure these in the `training.memory` section of your config file.

## Distributed Training

For multi-GPU training:

```bash
torchrun --nproc_per_node=NUM_GPUS src/main.py --config my_config.yaml
```

## License

MIT License - See LICENSE file for details.
