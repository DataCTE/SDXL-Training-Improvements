"""Configuration management for SDXL training."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_type: str = "sdxl"
    num_timesteps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    timestep_bias_strategy: str = "none"
    timestep_bias_min: float = 0.0
    timestep_bias_max: float = 1.0

    @property
    def kwargs(self) -> dict:
        """Get model configuration parameters."""
        return {
            "pretrained_model_name": self.pretrained_model_name,
            "model_type": self.model_type,
            "num_timesteps": self.num_timesteps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "timestep_bias_strategy": self.timestep_bias_strategy,
            "timestep_bias_min": self.timestep_bias_min,
            "timestep_bias_max": self.timestep_bias_max
        }

@dataclass
class OptimizerConfig:
    """Optimizer configuration with support for custom optimizers."""
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    optimizer_type: str = "adamw_bf16"
    
    # Schedule-free specific options
    warmup_steps: int = 0
    kahan_sum: bool = True
    correct_bias: bool = True
    
    # SOAP specific options
    precondition_frequency: int = 10
    shampoo_beta: float = 0.95
    max_precond_dim: int = 10000
    precondition_1d: bool = False
    merge_dims: bool = False
    normalize_grads: bool = False
    data_format: str = "channels_first"

    @property
    def class_name(self) -> str:
        """Map optimizer_type to optimizer class."""
        optimizer_map = {
            "adamw_bf16": "AdamWBF16",
            "adamw_schedule_free_kahan": "AdamWScheduleFreeKahan",
            "soap": "SOAP"
        }
        if self.optimizer_type.lower() not in optimizer_map:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        return optimizer_map[self.optimizer_type.lower()]

    @property
    def kwargs(self) -> dict:
        """Get optimizer configuration parameters based on type."""
        # Base parameters for all optimizers
        base_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (self.beta1, self.beta2),
            "eps": self.epsilon
        }
        
        # Add specific parameters based on optimizer type
        if self.optimizer_type == "adamw_bf16":
            # AdamWBF16 only uses base parameters
            return base_kwargs
            
        elif self.optimizer_type == "adamw_schedule_free_kahan":
            return {
                **base_kwargs,
                "warmup_steps": self.warmup_steps,
                "kahan_sum": self.kahan_sum,
                "correct_bias": self.correct_bias
            }
            
        elif self.optimizer_type == "soap":
            return {
                **base_kwargs,
                "correct_bias": self.correct_bias,
                "precondition_frequency": self.precondition_frequency,
                "shampoo_beta": self.shampoo_beta,
                "max_precond_dim": self.max_precond_dim,
                "precondition_1d": self.precondition_1d,
                "merge_dims": self.merge_dims,
                "normalize_grads": self.normalize_grads,
                "data_format": self.data_format
            }
            
        return base_kwargs

@dataclass
class SchedulerConfig:
    """Noise scheduler configuration."""
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    clip_sample: bool = False
    steps_offset: int = 0
    timestep_spacing: str = "leading"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    rescale_betas_zero_snr: bool = True

    @property
    def kwargs(self) -> dict:
        """Get scheduler configuration parameters."""
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "clip_sample": self.clip_sample,
            "steps_offset": self.steps_offset,
            "timestep_spacing": self.timestep_spacing,
            "thresholding": self.thresholding,
            "dynamic_thresholding_ratio": self.dynamic_thresholding_ratio,
            "sample_max_value": self.sample_max_value,
            "rescale_betas_zero_snr": self.rescale_betas_zero_snr
        }

@dataclass
class MethodConfig:
    """Training method configuration."""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 100
    save_every: int = 1
    method: str = "ddpm"  # or "flow_matching"
    prediction_type: str = "epsilon" # epsilon, v_prediction
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 4  # Fixed to 4 steps for stable training
    mixed_precision: str = "fp16"  # or "bf16", "no"
    enable_xformers: bool = True
    clip_grad_norm: float = 1.0
    num_inference_steps: int = 50
    method_config: MethodConfig = field(default_factory=MethodConfig)
    debug_mode: bool = False
    save_final_model: bool = True

    @property
    def dataloader_kwargs(self) -> dict:
        """Get DataLoader configuration."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": True,
            "drop_last": True,  # Important for stable training
            "persistent_workers": True if self.num_workers > 0 else False
        }

@dataclass
class ImageConfig:
    """Image processing configuration."""
    supported_dims: List[List[int]] = field(default_factory=lambda: [
        [1024, 1024],  # Square
        [1024, 1536],  # Portrait
        [1536, 1024],  # Landscape
    ])
    max_aspect_ratio: float = 2.0
    target_size: List[int] = field(default_factory=lambda: [1024, 1024])
    max_size: List[int] = field(default_factory=lambda: [1536, 1536])
    min_size: List[int] = field(default_factory=lambda: [512, 512])
    bucket_step: int = 64

@dataclass
class CacheConfig:
    cache_dir: Union[str, Path] = "cache"
    max_cache_size: int = 10000
    use_cache: bool = True
    cache_latents: bool = True
    cache_text_embeddings: bool = True

    @property
    def kwargs(self) -> dict:
        """Get cache configuration parameters."""
        return {
            "cache_dir": self.cache_dir,
            "max_cache_size": self.max_cache_size,
            "use_cache": self.use_cache,
            "cache_latents": self.cache_latents,
            "cache_text_embeddings": self.cache_text_embeddings
        }

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "sdxl-training"
    wandb_entity: Optional[str] = None
    log_dir: str = "logs"
    filename: str = "training.log"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    capture_warnings: bool = True
    log_every: int = 10

    @property
    def kwargs(self) -> dict:
        """Get logging configuration parameters."""
        return {
            "log_dir": self.log_dir,
            "filename": self.filename,
            "console_level": self.console_level,
            "file_level": self.file_level,
            "capture_warnings": self.capture_warnings,
            "wandb": {
                "use_wandb": self.use_wandb,
                "project": self.wandb_project,
                "entity": self.wandb_entity
            }
        }

@dataclass
class DataConfig:
    train_data_dir: Union[str, List[str]] = field(default_factory=lambda: ["data/train"])
    validation_data_dir: Optional[Union[str, List[str]]] = None
    image_size: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    tokenizer_max_length: int = 77

    @property
    def kwargs(self) -> dict:
        """Get data configuration parameters."""
        return {
            "train_data_dir": self.train_data_dir,
            "validation_data_dir": self.validation_data_dir,
            "image_size": self.image_size,
            "center_crop": self.center_crop,
            "random_flip": self.random_flip,
            "tokenizer_max_length": self.tokenizer_max_length
        }

@dataclass
class GlobalConfig:
    """Global configuration settings."""
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

@dataclass
class TagWeightingConfig:
    """Configuration for tag weighting."""
    enable_tag_weighting: bool = False
    use_cache: bool = True
    min_weight: float = 0.1
    max_weight: float = 3.0
    default_weight: float = 1.0
    smoothing_factor: float = 0.05

    @property
    def kwargs(self) -> dict:
        """Get tag weighting configuration parameters."""
        return {
            "enable_tag_weighting": self.enable_tag_weighting,
            "use_cache": self.use_cache,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "default_weight": self.default_weight,
            "smoothing_factor": self.smoothing_factor
        }

@dataclass
class Config:
    """Main configuration class for SDXL training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file with proper fallback hierarchy."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using default values")
            return cls()
            
        try:
            # Load raw YAML
            with open(path) as f:
                raw_config = yaml.safe_load(f)
            
            # Create default config as fallback
            config = cls()
            
            # Helper function to update config with YAML values
            def update_config(config_obj, yaml_data):
                if not yaml_data:
                    return config_obj
                
                # Get default values
                default_values = asdict(config_obj)
                
                # Update only values that exist in YAML
                for key, value in yaml_data.items():
                    if key in default_values:
                        if isinstance(value, dict) and hasattr(config_obj, key):
                            # Recursively update nested configs
                            nested_obj = getattr(config_obj, key)
                            if hasattr(nested_obj, '__dict__'):
                                setattr(config_obj, key, update_config(nested_obj, value))
                        else:
                            setattr(config_obj, key, value)
                
                return config_obj

            # Update each config section, maintaining defaults for missing values
            if 'model' in raw_config:
                config.model = update_config(config.model, raw_config['model'])
            
            if 'optimizer' in raw_config:
                config.optimizer = update_config(config.optimizer, raw_config['optimizer'])
            
            if 'training' in raw_config:
                training_data = raw_config['training'].copy()
                if 'method_config' in training_data:
                    scheduler_data = training_data['method_config'].get('scheduler', {})
                    config.training.method_config.scheduler = update_config(
                        config.training.method_config.scheduler,
                        scheduler_data
                    )
                    del training_data['method_config']
                config.training = update_config(config.training, training_data)
            
            if 'data' in raw_config:
                config.data = update_config(config.data, raw_config['data'])
            
            if 'global_config' in raw_config:
                global_data = raw_config['global_config']
                if 'cache' in global_data:
                    config.global_config.cache = update_config(
                        config.global_config.cache,
                        global_data['cache']
                    )
                if 'logging' in global_data:
                    config.global_config.logging = update_config(
                        config.global_config.logging,
                        global_data['logging']
                    )
                if 'image' in global_data:
                    config.global_config.image = update_config(
                        config.global_config.image,
                        global_data['image']
                    )
            
            if 'tag_weighting' in raw_config:
                config.tag_weighting = update_config(
                    config.tag_weighting,
                    raw_config['tag_weighting']
                )
            
            # Log which values came from YAML vs defaults
            if logger.isEnabledFor(logging.DEBUG):
                yaml_keys = set(_flatten_dict(raw_config).keys())
                default_keys = set(_flatten_dict(asdict(cls())).keys())
                logger.debug(f"Values from YAML: {yaml_keys}")
                logger.debug(f"Using defaults for: {default_keys - yaml_keys}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {str(e)}", exc_info=True)
            raise

def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Helper function to flatten nested dictionaries for logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
