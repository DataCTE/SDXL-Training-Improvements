"""Configuration management for SDXL training."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
        
    @dataclass 
    class CacheConfig:
        """Caching configuration."""
        cache_dir: str = "cache"
        use_cache: bool = True
        clear_cache_on_start: bool = False
        
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
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    num_epochs: int = 100
    warmup_steps: int = 500
    save_steps: int = 500
    log_steps: int = 10
    eval_steps: int = 100
    max_train_steps: Optional[int] = None
    lr_scheduler: str = "cosine"
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    optimizer_eps: float = 1e-8
    prediction_type: str = "epsilon"
    snr_gamma: Optional[float] = 5.0
    use_wandb: bool = True
    random_flip: bool = True
    center_crop: bool = True
    
@dataclass
class DataConfig:
    """Data configuration."""
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
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
        # Create output and cache directories
        Path(self.global_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.global_config.cache.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate image sizes
        max_h, max_w = self.global_config.image.max_size
        min_h, min_w = self.global_config.image.min_size
        if max_h < min_h or max_w < min_w:
            raise ValueError("max_size must be greater than min_size")
            
        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        # Validate batch sizes
        if self.training.batch_size < self.training.gradient_accumulation_steps:
            raise ValueError("batch_size must be >= gradient_accumulation_steps")
