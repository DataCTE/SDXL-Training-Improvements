"""Configuration management for SDXL training."""
from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import yaml
from omegaconf import OmegaConf

@dataclass
class ModelConfig:
    pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_type: str = "sdxl"
    prediction_type: str = "epsilon"  # or "v_prediction"
    num_timesteps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    timestep_bias_strategy: str = "none"
    timestep_bias_min: float = 0.0
    timestep_bias_max: float = 1.0

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    optimizer_type: str = "adamw"  # or "adamw_bf16", "adamw_8bit", "lion", "prodigy"

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 100
    save_every: int = 1
    method: str = "ddpm"  # or "flow_matching"
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # or "bf16", "no"
    enable_xformers: bool = True
    clip_grad_norm: float = 1.0
    num_inference_steps: int = 50

@dataclass
class CacheConfig:
    cache_dir: Union[str, Path] = "cache"
    max_cache_size: int = 10000
    enable_cache: bool = True
    cache_latents: bool = True
    cache_text_embeddings: bool = True

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

@dataclass
class DataConfig:
    train_data_dir: Union[str, List[str]] = field(default_factory=lambda: ["data/train"])
    validation_data_dir: Optional[Union[str, List[str]]] = None
    image_size: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    tokenizer_max_length: int = 77

@dataclass
class GlobalConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

@dataclass
class Config:
    """Main configuration class for SDXL training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        # Load YAML with OmegaConf for better interpolation support
        yaml_config = OmegaConf.load(path)
        
        # Convert to dict and create config
        config_dict = OmegaConf.to_container(yaml_config, resolve=True)
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": vars(self.model),
            "optimizer": vars(self.optimizer),
            "training": vars(self.training),
            "data": vars(self.data),
            "global_config": {
                "cache": vars(self.global_config.cache),
                "logging": vars(self.global_config.logging)
            }
        }

    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
