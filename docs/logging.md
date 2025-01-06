# Logging System Documentation

The SDXL Training Improvements framework provides a comprehensive logging system that handles console output, file logging, metrics tracking, progress bars, and optional Weights & Biases integration.

## Quick Start

```python
from src.core.logging import setup_logging

# Basic usage
logger = setup_logging(module_name="my_module")
logger.info("Starting process...")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")

# With progress tracking
logger.start_progress(total=100, desc="Processing data")
for i in range(100):
    # Do work
    logger.update_progress(1, batch_size=32)

# Log metrics
logger.log_metrics({
    "loss": 0.5,
    "accuracy": 0.95
})

# Track GPU memory
logger.log_memory()

# Cleanup
logger.finish()
```

## Configuration

The logging system can be configured using the `LogConfig` class:

```python
from src.core.logging import LogConfig, setup_logging

config = LogConfig(
    name="my_module",                    # Logger name
    log_dir="outputs/logs",             # Directory for log files
    console_level="INFO",               # Console logging level
    file_level="DEBUG",                 # File logging level
    filename="training.log",            # Log file name
    
    # Feature flags
    enable_console=True,                # Enable console output
    enable_file=True,                   # Enable file logging
    enable_wandb=False,                 # Enable W&B logging
    enable_progress=True,               # Enable progress tracking
    enable_metrics=True,                # Enable metrics tracking
    enable_memory=True,                 # Enable memory tracking
    capture_warnings=True,              # Capture Python warnings
    propagate=False,                    # Propagate to parent loggers
    
    # W&B settings
    wandb_project="my_project",         # W&B project name
    wandb_name=None,                    # W&B run name
    wandb_tags=[],                      # W&B tags
    
    # Progress settings
    progress_window=100,                # Window size for rate calculation
    progress_smoothing=0.3,             # Smoothing factor for EMA
    progress_position=0,                # Progress bar position
    progress_leave=True,                # Keep progress bar after completion
    
    # Metrics settings
    metrics_window=100,                 # Window size for metrics averaging
    metrics_history=True,               # Keep full metrics history
    metrics_prefix="metrics/"           # Prefix for metric names
)

logger = setup_logging(config=config)
```

## Advanced Usage

### Centralized Logging Management

For applications that need multiple loggers, use the `LogManager`:

```python
from src.core.logging import LogManager

# Get singleton instance
manager = LogManager.get_instance()

# Configure all loggers
manager.configure_from_config(config)

# Get logger by name
logger1 = manager.get_logger("module1")
logger2 = manager.get_logger("module2")

# Cleanup all loggers
manager.cleanup()
```

### Progress Tracking

The progress tracking system provides throughput metrics:

```python
# Start progress tracking
logger.start_progress(total=1000, desc="Training")

# Update with batch size for accurate throughput
for batch in dataloader:
    # Do work
    logger.update_progress(1, batch_size=len(batch))
    
    # Progress metrics are automatically logged:
    # - throughput/samples_per_sec
    # - throughput/batch_time_ms
    # - throughput/accumulated_samples
```

### Metrics Tracking

The metrics system supports windowed averaging and history:

```python
# Log single metrics
logger.log_metrics({
    "loss/train": 0.5,
    "accuracy/train": 0.95
})

# Log multiple metrics
logger.log_metrics({
    "loss/train": 0.5,
    "loss/val": 0.6,
    "accuracy/train": 0.95,
    "accuracy/val": 0.93,
    "learning_rate": 0.001
})

# Metrics are automatically:
# - Averaged over the configured window
# - Stored in history if enabled
# - Logged to W&B if enabled
```

### Memory Tracking

Track GPU memory usage:

```python
# Log current GPU memory
logger.log_memory()

# Logs the following metrics:
# - memory/gpu_allocated_gb
# - memory/gpu_reserved_gb
```

### Context Manager

The logger can be used as a context manager:

```python
with setup_logging(config=config) as logger:
    logger.info("Starting...")
    # Do work
    # Logger is automatically cleaned up after the block
```

## Implementation Details

The logging system is organized into several modules:

- `base.py`: Configuration classes and exceptions
- `formatters.py`: Console output formatting with colors
- `progress.py`: Progress tracking with throughput metrics
- `metrics.py`: Metrics tracking with windowed averaging
- `core.py`: Main UnifiedLogger implementation
- `logging.py`: High-level setup and management

Key features:

1. **Thread Safety**: All shared resources are protected by locks
2. **Resource Management**: Proper cleanup of file handlers and progress bars
3. **Extensibility**: Modular design allows easy addition of new features
4. **Performance**: Efficient metrics tracking with windowed averaging
5. **Integration**: Optional W&B integration for experiment tracking

## Best Practices

1. **Use Descriptive Names**: Use hierarchical names for metrics (e.g., "loss/train", "loss/val")
2. **Cleanup Resources**: Always call `finish()` or use context manager
3. **Configure Early**: Set up logging at the start of your application
4. **Use Appropriate Levels**: Use DEBUG for detailed info, INFO for progress, WARNING for issues, ERROR for failures
5. **Track Memory**: Regularly track memory usage in training loops
6. **Batch Metrics**: Log related metrics together in one call to `log_metrics()`