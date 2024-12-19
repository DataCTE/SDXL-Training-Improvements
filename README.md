# SDXL Training Improvements

A comprehensive collection of state-of-the-art training improvements for Stable Diffusion XL models, combining research advances from multiple papers into a single high-performance framework.

## Features & Implemented Papers

### Memory Optimization
- Gradient checkpointing and layer offloading [1]
- Mixed precision training with dynamic scaling [2]
- Efficient tensor memory management [3]

### Training Methods
- Flow matching for improved convergence [4]
- Dynamic tag-based loss weighting [5]
- Advanced noise scheduling with Karras sigmas [6]
- NovelAI V3 training improvements [7]

### Data Processing
- High-throughput preprocessing pipeline
- Aspect ratio bucketing for SDXL [8]
- Advanced caption preprocessing [9]

### Architecture
- Distributed training support
- CUDA-optimized operations
- Wandb integration for experiment tracking

## Paper Citations

[1] Gradient Checkpointing paper citation
[2] Mixed Precision Training paper citation
[3] Memory Management paper citation
[4] Flow Matching paper citation
[5] Dynamic Loss paper citation
[6] Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
[7] Ossa et al. "Improvements to SDXL in NovelAI Diffusion V3" arXiv:2312.12559, 2023
[8] Aspect Ratio Bucketing paper citation
[9] Caption Processing paper citation

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
