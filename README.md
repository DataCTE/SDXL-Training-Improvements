# SDXL Training Improvements

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive collection of state-of-the-art training improvements for Stable Diffusion XL models, combining research advances from multiple papers into a single high-performance framework.

## Features & Implemented Papers

### Memory Optimization
- Gradient checkpointing and layer offloading [[1]](#references)
- Mixed precision training with dynamic scaling [[2]](#references)
- Efficient tensor memory management [[3]](#references)

### Training Methods
- Flow matching with logit-normal sampling from nyaflow-xl [[4]](#references)
- Dynamic tag-based loss weighting [[5]](#references)
- Advanced noise scheduling with Karras sigmas [[6]](#references)
- NovelAI V3 training improvements [[7]](#references)

### Data Processing
- High-throughput preprocessing pipeline
- Aspect ratio bucketing for SDXL [[8]](#references)
- Advanced caption preprocessing [[9]](#references)

### Architecture
- Distributed training support
- CUDA-optimized operations
- Weights & Biases integration for experiment tracking

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

1. Gradient Checkpointing paper citation
2. Mixed Precision Training paper citation
3. Memory Management paper citation
4. [nyaflow-xl-alpha: SDXL finetuning with Flow Matching](https://huggingface.co/nyanko7/nyaflow-xl-alpha)
5. Dynamic Loss paper citation
6. Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
7. Ossa et al. ["Improvements to SDXL in NovelAI Diffusion V3"](https://arxiv.org/abs/2312.12559)
8. Aspect Ratio Bucketing paper citation
9. Caption Processing paper citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
