# SDXL Training Framework with Novel Research Methods

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-focused SDXL training framework implementing cutting-edge advances in diffusion model training, with emphasis on image quality and training stability.

## Training Methods

### Flow Matching with Logit-Normal Sampling [[4]](#references)

Advanced training method that eliminates noise scheduling:

```python
# Configure Flow Matching training
training:
  method: "flow_matching"
  batch_size: 4
  learning_rate: 1.0e-6
```

Key benefits:
- 30% faster convergence via optimal transport paths
- Direct velocity field learning reduces instability
- No noise schedule dependencies
- Logit-normal time sampling for better coverage

### NovelAI V3 UNet Architecture [[7]](#references)

State-of-the-art model improvements:

```python
# Enable NovelAI V3 features
training:
  prediction_type: "v_prediction"
  zero_terminal_snr: true
  sigma_max: 20000.0
```

Improvements:
- Zero Terminal SNR training
  - Infinite noise approximation (σ_max ≈ 20000)
  - Better artifact handling
  - Enhanced detail preservation
- V-prediction parameterization
  - Stable high-frequency gradients
  - Reduced color shifting
- Dynamic Karras schedule
  - Adaptive noise spacing
  - Improved texture quality

## Image Quality Improvements

- Enhanced Detail Preservation
  - Fine-grained texture generation
  - Improved handling of complex patterns
  - Better preservation of small objects and features

- Color and Lighting
  - More accurate color reproduction
  - Enhanced dynamic range in highlights and shadows
  - Better handling of complex lighting conditions

- Composition and Structure
  - Improved spatial coherence
  - Better handling of perspective and depth
  - More consistent object proportions


## Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.8+ |
| CUDA      | 11.7+ |
| VRAM      | 24GB+ |

## Installation

```bash
# Clone repository
git clone https://github.com/DataCTE/SDXL-Training-Improvements.git
cd SDXL-Training-Improvements

# Install in development mode with all extras
pip install -e ".[dev,docs]"

# Verify installation
python -c "import src; print(src.__version__)"
```

## Configuration

The training framework is configured through a YAML file. Key configuration sections:

```yaml
# Model configuration
model:
  pretrained_model_name: "stabilityai/stable-diffusion-xl-base-1.0"
  num_timesteps: 1000
  sigma_min: 0.002
  sigma_max: 80.0

# Training parameters  
training:
  batch_size: 4
  learning_rate: 4.0e-7
  method: "ddpm"  # or "flow_matching"
  zero_terminal_snr: true
  
# Dataset configuration
data:
  train_data_dir: 
    - "path/to/dataset1"
    - "path/to/dataset2"
```

See [config.yaml](src/config.yaml) for full configuration options.

## Usage Examples

Basic training:
```bash
# Train with default config
python src/main.py --config config.yaml

# Train with custom config
python src/main.py --config my_config.yaml

# Distributed training
torchrun --nproc_per_node=2 src/main.py --config config.yaml
```

## References

4. nyanko7, "nyaflow-xl-alpha: SDXL finetuning with Flow Matching", https://huggingface.co/nyanko7/nyaflow-xl-alpha, 2024
7. Ossa et al., "Improvements to SDXL in NovelAI Diffusion V3", arXiv:2409.15997, 2024

## License

MIT License - see [LICENSE](LICENSE) file.
