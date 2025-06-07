# SDXL Training Framework

A streamlined and optimized framework for fine-tuning Stable Diffusion XL models with state-of-the-art techniques.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/DataCTE/SDXL-Training-Improvements.git
cd SDXL-Training-Improvements

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your dataset with the following structure:
```
data/train/
├── image1.jpg
├── image1.txt  # Caption for image1
├── image2.png
├── image2.txt  # Caption for image2
└── ...
```

### 3. Configure Training

Edit `config.yaml` to customize your training:

```yaml
model:
  name: "stabilityai/stable-diffusion-xl-base-1.0"
  dtype: "bf16"  # Use bf16 for better stability

training:
  batch_size: 1  # Adjust based on your GPU memory
  gradient_accumulation_steps: 4
  learning_rate: 1e-6
  num_epochs: 100
  
dataset:
  train_dir: "./data/train"  # Path to your dataset
```

### 4. Start Training

```bash
# Basic training
python train.py

# With custom config
python train.py --config my_config.yaml

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint-500
```

## Key Features

### 🚀 Performance Optimizations
- **Mixed Precision Training**: Automatic bf16/fp16 support for faster training
- **Gradient Checkpointing**: Reduce memory usage by up to 60%
- **xFormers Integration**: Efficient attention mechanisms
- **Multi-GPU Support**: Built-in distributed training with Accelerate

### 🎯 Training Improvements
- **V-Prediction**: Better color accuracy and reduced artifacts
- **Zero Terminal SNR**: Improved detail preservation
- **Aspect Ratio Bucketing**: Train on multiple resolutions efficiently
- **Advanced Schedulers**: Cosine annealing with warmup

### 📊 Monitoring
- **Weights & Biases Integration**: Real-time metrics tracking
- **Automatic Checkpointing**: Never lose progress
- **Rich Console Output**: Beautiful progress bars and status updates

## Configuration Options

### Model Settings
```yaml
model:
  name: "model_id_or_path"  # HuggingFace model ID or local path
  dtype: "bf16"             # fp32, fp16, or bf16
```

### Training Parameters
```yaml
training:
  batch_size: 1                    # Images per GPU
  gradient_accumulation_steps: 4   # Effective batch = batch_size * grad_accum * num_gpus
  learning_rate: 1e-6              # Starting learning rate
  num_epochs: 100                  # Total training epochs
  
  # Checkpointing
  save_every_n_steps: 500          # Save frequency
  checkpoint_dir: "./checkpoints"   # Save location
  
  # Memory optimization
  gradient_checkpointing: true     # Trade compute for memory
  use_xformers: true               # Efficient attention
  
  # Advanced options
  method: "ddpm"                   # or "flow_matching" (experimental)
```

### Dataset Configuration
```yaml
dataset:
  train_dir: "./data/train"        # Training data location
  image_size: 1024                 # Base resolution
  center_crop: true                # Center crop images
  random_flip: true                # Horizontal flip augmentation
  
  # Multi-aspect training
  use_aspect_ratio_bucketing: true # Group by aspect ratio
  min_size: 512                    # Minimum dimension
  max_size: 1536                   # Maximum dimension
```

### Advanced Settings (NovelAI V3)
```yaml
advanced:
  v_prediction: true               # V-parameterization
  zero_terminal_snr: true          # Zero SNR at t=T
  sigma_max: 20000.0               # Maximum noise level
  min_snr_gamma: 5.0               # SNR loss weighting
```

## Multi-GPU Training

For distributed training across multiple GPUs:

```bash
# Single node, multiple GPUs
accelerate launch train.py

# Custom configuration
accelerate config  # Follow the prompts
accelerate launch train.py
```

## Memory Requirements

| Resolution | Batch Size | VRAM Usage (bf16) | Recommended GPU |
|------------|-----------|-------------------|-----------------|
| 512x512    | 4         | ~8GB              | RTX 3070        |
| 768x768    | 2         | ~12GB             | RTX 3080        |
| 1024x1024  | 1         | ~16GB             | RTX 4080        |
| 1024x1024  | 4         | ~24GB             | RTX 4090        |

## Tips for Best Results

1. **Dataset Quality**: 
   - Use high-quality images (at least 1024x1024)
   - Write detailed, accurate captions
   - Aim for at least 100-1000 images for good results

2. **Learning Rate**:
   - Start with 1e-6 for fine-tuning
   - Use 5e-7 for smaller datasets
   - Monitor loss curves and adjust accordingly

3. **Training Duration**:
   - Most models converge within 50-200 epochs
   - Save checkpoints frequently to find the best version
   - Watch for overfitting on small datasets

4. **Memory Optimization**:
   - Enable gradient checkpointing for larger batch sizes
   - Use bf16 instead of fp32 for 2x memory savings
   - Reduce batch size and increase gradient accumulation

## Troubleshooting

### Out of Memory
```yaml
# Reduce memory usage
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

### Slow Training
- Enable xformers: `use_xformers: true`
- Use bf16: `mixed_precision: "bf16"`
- Reduce image size temporarily
- Check GPU utilization with `nvidia-smi`

### Poor Quality Results
- Increase dataset size and quality
- Lower learning rate
- Train for more epochs
- Try v-prediction mode

## License

MIT License - see [LICENSE](LICENSE) file for details.
