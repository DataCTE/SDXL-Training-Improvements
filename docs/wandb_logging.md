# Weights & Biases Logging

The SDXL Training Improvements framework provides a comprehensive Weights & Biases (W&B) logging utility through the `WandbLogger` class.

## Quick Start

```python
from src.core.logging import WandbLogger, WandbConfig, UnifiedLogger

# Create W&B logger
wandb_config = WandbConfig(
    project="my-sdxl-project",
    name="experiment-1",
    tags=["sdxl", "training"],
    log_model=True
)

# Optional: Use with UnifiedLogger
logger = UnifiedLogger()

# Initialize W&B logger
with WandbLogger(config=wandb_config, logger=logger) as wandb_logger:
    # Log metrics
    wandb_logger.log_metrics({
        "loss": 0.5,
        "accuracy": 0.95
    }, step=100)
    
    # Log images
    wandb_logger.log_images(
        images=[generated_image],
        captions=["Generated sample at step 100"]
    )
    
    # Log model checkpoint
    wandb_logger.log_model(
        "checkpoints/model.pt",
        metadata={"step": 100},
        aliases=["latest"]
    )
```

## Configuration

The `WandbConfig` class provides extensive configuration options:

```python
from src.core.logging import WandbConfig

config = WandbConfig(
    # Basic settings
    project="sdxl-training",      # W&B project name
    name="experiment-1",          # Run name
    tags=["sdxl", "training"],    # Run tags
    notes="Training notes",       # Run notes
    group="experiment-group",     # Group for related runs
    job_type="training",         # Type of job
    
    # Run settings
    dir="outputs/wandb",         # Directory for W&B files
    resume=False,                # Resume previous run
    reinit=False,               # Allow multiple runs in same process
    mode="online",              # "online", "offline", or "disabled"
    
    # Sync settings
    sync_tensorboard=False,      # Sync TensorBoard logs
    save_code=True,             # Save code snapshot
    save_requirements=True,      # Save pip requirements
    
    # Media settings
    log_model=True,             # Log model checkpoints
    log_checkpoints=True,       # Log intermediate checkpoints
    checkpoint_prefix="ckpt",   # Prefix for checkpoint artifacts
    sample_prefix="sample",     # Prefix for image samples
    max_images_to_log=16,       # Max images per batch
    
    # Metric settings
    metric_prefix="metrics/",   # Prefix for metric names
    log_system_metrics=True,    # Log GPU/CPU metrics
    system_metrics_interval=60  # Seconds between system metrics
)
```

## Logging Features

### Metrics Logging

```python
# Log single metrics
wandb_logger.log_metrics({
    "loss/train": 0.5,
    "accuracy/train": 0.95
}, step=100)

# Log multiple metrics
wandb_logger.log_metrics({
    "loss/train": 0.5,
    "loss/val": 0.6,
    "accuracy/train": 0.95,
    "accuracy/val": 0.93,
    "learning_rate": 0.001
}, step=100)

# System metrics are logged automatically if enabled
# - system/gpu0/memory_allocated
# - system/gpu0/memory_reserved
# - system/gpu0/utilization
# - system/cpu/percent
# - system/memory/percent
# - system/disk/percent
```

### Image Logging

```python
# Log PIL images
from PIL import Image
image = Image.open("sample.png")
wandb_logger.log_images(
    images=[image],
    captions=["Sample image"]
)

# Log torch tensors
import torch
tensor = torch.randn(3, 64, 64)  # [C, H, W]
wandb_logger.log_images(
    images=[tensor],
    prefix="generated"  # Will be logged as "generated/images"
)

# Log numpy arrays
import numpy as np
array = np.random.rand(64, 64, 3)  # [H, W, C]
wandb_logger.log_images(
    images=[array],
    captions=["Random noise"]
)

# Log multiple images
wandb_logger.log_images(
    images=[img1, img2, img3],
    captions=["Sample 1", "Sample 2", "Sample 3"],
    step=100
)
```

### Model Logging

```python
# Log model checkpoint
wandb_logger.log_model(
    "checkpoints/model.pt",
    metadata={
        "step": 1000,
        "loss": 0.5,
        "config": model_config
    },
    aliases=["latest", "best"]
)

# Log intermediate checkpoint
wandb_logger.log_model(
    "checkpoints/step_1000.pt",
    metadata={"step": 1000},
    aliases=["step-1000"]
)
```

### Configuration Logging

```python
# Log training configuration
wandb_logger.log_config({
    "model": {
        "type": "sdxl",
        "version": "1.0",
        "params": 1e9
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100
    },
    "data": {
        "dataset": "custom",
        "size": 10000
    }
})
```

## Integration with UnifiedLogger

The `WandbLogger` integrates seamlessly with the `UnifiedLogger`:

```python
from src.core.logging import UnifiedLogger, WandbLogger, LogConfig, WandbConfig

# Create loggers
log_config = LogConfig(name="training")
wandb_config = WandbConfig(project="sdxl")

logger = UnifiedLogger(log_config)
wandb_logger = WandbLogger(wandb_config, logger=logger)

# Use both loggers
with logger, wandb_logger:
    # Regular logging goes to both console and file
    logger.info("Starting training...")
    
    # Progress tracking
    logger.start_progress(total=100)
    for step in range(100):
        # Update progress
        logger.update_progress(1)
        
        # Log metrics to both W&B and local
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95
        }
        logger.log_metrics(metrics)
        wandb_logger.log_metrics(metrics, step=step)
        
        # Log images to W&B
        if step % 10 == 0:
            wandb_logger.log_images(
                images=[generate_sample()],
                step=step
            )
            
        # Log model checkpoints
        if step % 50 == 0:
            wandb_logger.log_model(
                f"checkpoints/step_{step}.pt",
                metadata={"step": step}
            )
```

## Best Practices

1. **Use Consistent Prefixes**: Keep metric names organized with prefixes (e.g., "loss/train", "loss/val")
2. **Log System Metrics**: Enable `log_system_metrics` to track resource usage
3. **Use Context Manager**: The context manager ensures proper cleanup
4. **Add Metadata**: Include relevant metadata with model checkpoints
5. **Group Related Runs**: Use the `group` parameter to organize related experiments
6. **Add Tags**: Use tags to categorize runs for easier filtering
7. **Save Code**: Enable `save_code` to track code changes between runs
8. **Limit Images**: Use `max_images_to_log` to prevent memory issues
9. **Resume Runs**: Use `resume=True` to continue crashed runs
10. **Use Aliases**: Add aliases to model artifacts for easy reference