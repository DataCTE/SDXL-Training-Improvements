"""Configuration management for SDXL training."""
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import yaml
from src.core.logging.logging import setup_logging

logger = setup_logging(__name__)

@dataclass
class GlobalConfig:
    """Global configuration settings."""
    
    @dataclass
    class ImageConfig:
        """Image processing configuration."""
        target_size: Tuple[int, int] = (1024, 1024)
        supported_dims: List[Tuple[int, int]] = field(default_factory=lambda: [
            (1024, 1024),
            (1152, 896),
            (896, 1152),
            (1216, 832),
            (832, 1216),
            (1344, 768),
            (768, 1344),
            (1536, 640),
            (640, 1536)
        ])
        max_size: Tuple[int, int] = (1536, 1536)
        min_size: Tuple[int, int] = (640, 640)
        max_dim: int = 1536 * 1536  # Max total pixels
        bucket_step: int = 64  # Step size for bucketing dimensions
        max_aspect_ratio: float = 2.0  # Maximum allowed aspect ratio for images
        
    @dataclass 
    class CacheConfig:
        """Caching configuration."""
        cache_dir: str = "cache"
        use_cache: bool = True
        clear_cache_on_start: bool = False
        num_proc: int = 4  # Number of processes for cache operations
        chunk_size: int = 1000  # Number of items per cache chunk
        compression: str = "zstd"  # Compression algorithm (None, 'zstd', 'gzip')
        verify_hashes: bool = True  # Whether to verify content hashes
        
    image: ImageConfig = field(default_factory=ImageConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    seed: Optional[int] = None
    output_dir: str = "outputs"
    
@dataclass
class ModelConfig:
    """Model configuration."""
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_timesteps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    dtype: str = "bfloat16"  # Model precision: float32, float16, or bfloat16
    fallback_dtype: str = "float32"  # Fallback precision when main dtype not supported
    unet_dtype: Optional[str] = None  # UNet specific dtype, falls back to main dtype if None
    prior_dtype: Optional[str] = None  # Prior model dtype, falls back to main dtype if None
    
@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    enable_24gb_optimizations: bool = False
    layer_offload_fraction: float = 0.0
    enable_activation_offloading: bool = False
    enable_async_offloading: bool = True
    temp_device: str = "cpu"

@dataclass
class FlowMatchingConfig:
    """Flow Matching configuration."""
    enabled: bool = False
    num_timesteps: int = 1000
    sigma: float = 1.0
    time_sampling: str = "uniform"  # uniform, logit_normal

@dataclass 
class DDPMConfig:
    """DDPM-specific training configuration.
    
    Prediction Types:
    - "epsilon": Original DDPM noise prediction, more stable but slower convergence
    - "v_prediction": Velocity prediction (like NovelAI), faster convergence but needs tuning
    - "sample": Direct sample prediction, experimental
    
    Compatible Methods:
    - Works with: "ddpm", "ddim", "dpm-solver", "euler", "euler-ancestral"
    - Best results with "ddpm" for training, "euler-ancestral" for inference
    """
    prediction_type: str = "v_prediction"  # v_prediction, epsilon, or sample
    snr_gamma: Optional[float] = 5.0  # Signal-to-noise ratio gamma, None disables SNR weighting
    zero_terminal_snr: bool = True    # Enable zero terminal SNR for better quality
    sigma_min: float = 0.002         # Min noise level, lower = sharper but may be unstable
    sigma_max: float = 20000.0       # Max noise level, higher = more diversity
    rho: float = 7.0                 # Karras scheduler parameter

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    learning_rate: float = 4.0e-7
    max_grad_norm: float = 1.0
    num_epochs: int = 100
    warmup_steps: int = 500
    save_steps: int = 500
    log_steps: int = 10
    eval_steps: int = 100
    validation_steps: int = 1000
    validation_samples: int = 4
    validation_guidance_scale: float = 7.5
    validation_inference_steps: int = 30
    max_train_steps: Optional[int] = None
    lr_scheduler: str = "linear"
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    optimizer_eps: float = 1e-8
    use_wandb: bool = True
    random_flip: bool = True
    center_crop: bool = True
    method: str = "ddpm"  # ddpm or flow_matching
    prediction_type: str = "v_prediction"  # v_prediction, epsilon, or sample
    zero_terminal_snr: bool = True  # Enable zero terminal SNR for better quality
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    
@dataclass
class DataConfig:
    """Data configuration."""
    train_data_dir: Union[str, List[str]] = field(default_factory=lambda: ["data/train"])
    val_data_dir: Union[str, List[str]] = field(default_factory=lambda: ["data/val"])
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    proportion_empty_prompts: float = 0.05
    
@dataclass
class TagWeightingConfig:
    """Tag weighting configuration."""
    enable_tag_weighting: bool = True
    use_cache: bool = True
    default_weight: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 10.0
    smoothing_factor: float = 0.1

@dataclass
class Config:
    """Complete training configuration."""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        from .utils.paths import convert_windows_path
        # Create output and cache directories with WSL path handling
        output_dir = convert_windows_path(self.global_config.output_dir, make_absolute=True)
        cache_dir = convert_windows_path(self.global_config.cache.cache_dir, make_absolute=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate image sizes with detailed error tracking
        try:
            # Extract and validate size tuples
            max_size = tuple(int(x) for x in self.global_config.image.max_size) if isinstance(self.global_config.image.max_size, (list, tuple)) else self.global_config.image.max_size
            min_size = tuple(int(x) for x in self.global_config.image.min_size) if isinstance(self.global_config.image.min_size, (list, tuple)) else self.global_config.image.min_size
            
            # Convert to tuple and store back
            self.global_config.image.max_size = max_size
            self.global_config.image.min_size = min_size
            
            # Validate tuple types
            if not isinstance(max_size, tuple):
                raise ValueError(f"max_size must be a tuple or list, got {type(max_size)}\nValue: {repr(max_size)}")
            if not isinstance(min_size, tuple):
                raise ValueError(f"min_size must be a tuple or list, got {type(min_size)}\nValue: {repr(min_size)}")
                
            # Validate tuple lengths
            if len(max_size) != 2:
                raise ValueError(f"max_size must contain exactly 2 values, got {len(max_size)}\nValue: {repr(max_size)}")
            if len(min_size) != 2:
                raise ValueError(f"min_size must contain exactly 2 values, got {len(min_size)}\nValue: {repr(min_size)}")
                
            # Now safely unpack
            max_h, max_w = max_size
            min_h, min_w = min_size
            
            # Validate individual values
            for name, value in [
                ("max_h", max_h), ("max_w", max_w),
                ("min_h", min_h), ("min_w", min_w)
            ]:
                if not isinstance(value, int):
                    raise ValueError(
                        f"{name} must be an integer, got {type(value)}\n"
                        f"Value: {repr(value)}"
                    )
                    
            # Compare dimensions with detailed error
            if max_h < min_h or max_w < min_w:
                raise ValueError(
                    f"max_size dimensions must be greater than min_size dimensions\n"
                    f"max_size: {repr(max_size)}\n"
                    f"min_size: {repr(min_size)}\n"
                    f"Comparison: max_h ({max_h}) < min_h ({min_h}) or max_w ({max_w}) < min_w ({min_w})"
                )
                
            # Log actual values for debugging
            logger.debug(
                "Image size validation:\n"
                f"max_size: {repr(max_size)} ({type(max_size)})\n"
                f"min_size: {repr(min_size)} ({type(min_size)})\n"
                f"Individual values:\n"
                f"- max_h: {repr(max_h)} ({type(max_h)})\n"
                f"- max_w: {repr(max_w)} ({type(max_w)})\n"
                f"- min_h: {repr(min_h)} ({type(min_h)})\n"
                f"- min_w: {repr(min_w)} ({type(min_w)})"
            )
            
        except Exception as e:
            logger.error(
                f"Image size validation failed:\n"
                f"Error: {str(e)}\n"
                f"max_size: {repr(max_size)} ({type(max_size)})\n"
                f"min_size: {repr(min_size)} ({type(min_size)})\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
            
        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        # Validate batch sizes
        if self.training.batch_size < self.training.gradient_accumulation_steps:
            raise ValueError("batch_size must be >= gradient_accumulation_steps")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Loaded Config object
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Create nested dataclass instances
        global_config_dict = config_dict.get('global_config', {})
        global_config = GlobalConfig(
            image=GlobalConfig.ImageConfig(**global_config_dict.get('image', {})),
            cache=GlobalConfig.CacheConfig(**global_config_dict.get('cache', {})),
            seed=global_config_dict.get('seed'),
            output_dir=global_config_dict.get('output_dir', 'outputs')
        )
        
        model_config = ModelConfig(**config_dict.get('model', {}))
        
        training_dict = config_dict.get('training', {})
        training_config = TrainingConfig(
            memory=MemoryConfig(**training_dict.get('memory', {})),
            flow_matching=FlowMatchingConfig(**training_dict.get('flow_matching', {})),
            **{k: v for k, v in training_dict.items() if k not in ['memory', 'flow_matching']}
        )
        
        data_config = DataConfig(**config_dict.get('data', {}))
        tag_weighting_config = TagWeightingConfig(**config_dict.get('tag_weighting', {}))
        
        return cls(
            global_config=global_config,
            model=model_config,
            training=training_config,
            data=data_config,
            tag_weighting=tag_weighting_config
        )
