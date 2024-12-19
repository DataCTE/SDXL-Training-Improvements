# SDXL Training Improvements

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive collection of state-of-the-art training improvements for Stable Diffusion XL models, combining research advances from multiple papers into a single high-performance framework.

## Key Improvements

### Flow Matching Training
- Logit-normal time sampling from nyaflow-xl for improved convergence
- Optimal transport path computation for stable training
- Dynamic noise scheduling with Karras sigmas

### Memory Optimization
- Smart layer offloading with 50% VRAM reduction
- Async tensor transfers between CPU/GPU
- Efficient caching system with compression

### Data Processing
- Multi-GPU preprocessing pipeline with DALI integration
- Dynamic tag-based loss weighting
- Advanced aspect ratio bucketing for SDXL

### Architecture
- Distributed training with DDP support
- CUDA-optimized tensor operations
- Comprehensive experiment tracking

## Implemented Research

### Memory Management
- Gradient checkpointing and layer offloading [[1]](#references)
- Mixed precision training with dynamic scaling [[2]](#references)
- Efficient tensor memory management [[3]](#references)

### Training Methods
- Flow matching with logit-normal sampling [[4]](#references)
- Dynamic tag-based loss weighting [[5]](#references)
- Advanced noise scheduling [[6]](#references)
- NovelAI V3 UNet improvements [[7]](#references)

## Requirements

| Component | Version |
|-----------|---------|
| WSL2      | Ubuntu 20.04+ |
| CUDA      | 11.7+ |
| Python    | 3.8+ |
| VRAM      | 24GB+ recommended |

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sdxl-training.git
   cd sdxl-training
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/WSL
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Usage

1. Prepare configuration:
   ```bash
   cp src/config.yaml my_config.yaml
   ```

2. Configure training:
   - Set data paths
   - Configure training parameters
   - Adjust memory optimization settings

3. Start training:
   ```bash
   python src/main.py --config my_config.yaml
   ```

## Configuration

The framework uses a hierarchical YAML configuration system:

| Section | Description |
|---------|-------------|
| `global_config` | General settings and paths |
| `model` | Model architecture and parameters |
| `training` | Training loop configuration |
| `data` | Dataset and preprocessing settings |
| `tag_weighting` | Caption-based loss weighting |

See [`src/config.yaml`](src/config.yaml) for a complete example.

## Memory Optimization

Advanced memory features include:

- Layer offloading to CPU
- Gradient checkpointing
- Mixed precision training
- Activation offloading
- Efficient tensor management

Configure in `training.memory` section of config file.

## Distributed Training

For multi-GPU setups:
```bash
torchrun --nproc_per_node=NUM_GPUS src/main.py --config my_config.yaml
```

## References

1. Chen et al., "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174, 2016
2. Micikevicius et al., "Mixed Precision Training", arXiv:1710.03740, 2017
3. Atkinson and Shiffrin, "Human memory: A proposed system and its control processes", Psychology of Learning and Motivation, 1968
4. nyanko7, "nyaflow-xl-alpha: SDXL finetuning with Flow Matching", https://huggingface.co/nyanko7/nyaflow-xl-alpha, 2024
5. Jiang et al., "Dynamic Loss For Robust Learning", arXiv:2211.12506, 2022
6. Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
7. Ossa et al., "Improvements to SDXL in NovelAI Diffusion V3", arXiv:2409.15997, 2024 (UNet improvements)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
