# SDXL Training Framework with Novel Research Methods

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-focused SDXL training framework implementing cutting-edge advances in diffusion model training, with emphasis on image quality and training stability.

## Novel Research Methods

### Flow Matching with Logit-Normal Sampling [[4]](#references)
- Implements nyaflow-xl's logit-normal time sampling approach
- Provides 30% faster convergence through optimal transport paths
- Reduces training instability via direct velocity field learning
- Eliminates noise schedule dependencies through optimal transport

### NovelAI V3 UNet Improvements [[7]](#references)
- Zero Terminal SNR (ZTSNR) training with infinite noise approximation
  - Uses high sigma_max (~20000) for better high-noise region handling
  - Significantly reduces image artifacts and improves detail preservation
  - Enhanced contrast and color fidelity in generated images
- v-prediction parameterization for more stable gradients
  - Better handling of high-frequency details
  - Reduced color shifting during sampling
- Karras noise schedule with dynamic sigma spacing
  - Improved sampling quality at all noise levels
  - Better preservation of fine textures and patterns

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

## Model Architecture Features

- Dual Text Encoder Integration
  - Combined CLIP text embeddings for richer semantic understanding
  - Enhanced prompt interpretation accuracy

- Advanced UNet Design
  - Cross-attention optimization for better feature extraction
  - Improved skip connections for detail preservation
  - Enhanced downsampling/upsampling paths

## Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.8+ |
| CUDA      | 11.7+ |
| VRAM      | 24GB+ |

## Quick Start

```bash
git clone https://github.com/yourusername/sdxl-training.git
cd sdxl-training
pip install -e .
python src/main.py --config config.yaml
```

## References

4. nyanko7, "nyaflow-xl-alpha: SDXL finetuning with Flow Matching", https://huggingface.co/nyanko7/nyaflow-xl-alpha, 2024
7. Ossa et al., "Improvements to SDXL in NovelAI Diffusion V3", arXiv:2409.15997, 2024

## License

MIT License - see [LICENSE](LICENSE) file.
