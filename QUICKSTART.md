# Quick Start Guide

## 1. One-Line Setup (Linux/Mac)

```bash
chmod +x setup.sh && ./setup.sh
```

## 2. Prepare Your Dataset

Place your images and captions in the `data/train/` folder:
```
data/train/
├── cat.jpg
├── cat.txt      # "a fluffy orange cat sitting on a windowsill"
├── dog.png
├── dog.txt      # "a golden retriever playing in the park"
└── ...
```

## 3. Start Training

```bash
# Activate environment
source venv/bin/activate

# Start training
python train.py
```

## 4. Monitor Progress

Watch the console for real-time updates:
```
Epoch 1/100: 100%|████████| 250/250 [05:23<00:00, loss=0.0823]
[green]Saved checkpoint to checkpoints/checkpoint-500[/green]
```

## 5. Generate Images

After training:
```bash
# Generate with your trained model
python inference.py "a magical forest at sunset" --checkpoint checkpoints/checkpoint-500/unet

# Generate multiple variations
python inference.py "cyberpunk city" --num-images 4 --seed 42
```

## Common Commands

### Resume Training
```bash
python train.py --resume checkpoints/checkpoint-500
```

### Custom Configuration
```bash
python train.py --config my_config.yaml
```

### Multi-GPU Training
```bash
accelerate launch train.py
```

## Tips

- **Low VRAM?** Set `batch_size: 1` and `gradient_checkpointing: true`
- **Faster training?** Use `mixed_precision: "bf16"`
- **Better quality?** Train for more epochs and use high-quality data

## Need Help?

- Check `config.yaml` for all options
- Read the full README.md for detailed documentation
- Look at error messages - they usually tell you what's wrong!