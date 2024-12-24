"""Configuration management for SDXL training."""
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from src.core.logging.logging import setup_logging

logger = setup_logging(__name__)

@dataclass
class TransformsConfig:
    """Image transform configuration."""
    normalize: bool = True
    random_flip: bool = True
    center_crop: bool = True
    random_rotation: bool = False
    rotation_degrees: float = 5.0
    color_jitter: bool = False
    jitter_brightness: float = 0.1
    jitter_contrast: float = 0.1
    jitter_saturation: float = 0.1
    jitter_hue: float = 0.1

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
            (640, 1536),
        ])
        max_size: Tuple[int, int] = (1536, 1536)
        min_size: Tuple[int, int] = (640, 640)
        max_dim: int = 1536 * 1536  # Max total pixels
        bucket_step: int = 64       # Step size for bucketing dimensions
        max_aspect_ratio: float = 2.0
        min_aspect_ratio: float = 0.5
        resize_mode: str = "area"   # area, bilinear, bicubic, etc.
        
    @dataclass
    class CacheConfig:
        """Caching configuration."""
        cache_dir: str = "cache"
        use_cache: bool = True
        clear_cache_on_start: bool = False
        num_proc: int = 4
        chunk_size: int = 1000
        compression: str = "zstd"
        verify_hashes: bool = True
        max_memory_usage: float = 0.8
        enable_memory_tracking: bool = True
        cache_text_embeddings: bool = True
        cache_latents: bool = True
        cache_validation: bool = True
        
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
    dtype: str = "bfloat16"
    fallback_dtype: str = "float32"
    enable_bf16_training: bool = True
    unet_dtype: Optional[str] = None
    prior_dtype: Optional[str] = None
    text_encoder_dtype: Optional[str] = None
    text_encoder_2_dtype: Optional[str] = None
    vae_dtype: Optional[str] = None
    effnet_dtype: Optional[str] = None
    decoder_dtype: Optional[str] = None
    decoder_text_encoder_dtype: Optional[str] = None
    decoder_vqgan_dtype: Optional[str] = None
    lora_dtype: Optional[str] = None
    embedding_dtype: Optional[str] = None
    scheduler_type: str = "ddpm"
    scheduler_config: Optional[Dict] = None

@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    enable_24gb_optimizations: bool = False
    layer_offload_fraction: float = 0.0
    enable_activation_offloading: bool = False
    enable_async_offloading: bool = True
    temp_device: str = "cpu"
    max_memory_usage: float = 0.8
    enable_memory_tracking: bool = True
    offload_models: List[str] = field(default_factory=lambda: ["vae"])

@dataclass
class FlowMatchingConfig:
    """Flow Matching configuration."""
    enabled: bool = False
    num_timesteps: int = 1000
    sigma: float = 1.0
    time_sampling: str = "uniform"  # or logit_normal

@dataclass
class DDPMConfig:
    """DDPM-specific training configuration."""
    prediction_type: str = "v_prediction"
    snr_gamma: Optional[float] = 5.0
    zero_terminal_snr: bool = True
    sigma_min: float = 0.002
    sigma_max: float = 20000.0
    rho: float = 7.0

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adamw"  # adamw, adamw_bf16, adamw_kahan, soap
    soap_config: Dict = field(default_factory=lambda: {
        "precondition_frequency": 10,
        "max_precond_dim": 10000,
        "merge_dims": False,
        "precondition_1d": False,
        "normalize_grads": False
    })
    kahan_config: Dict = field(default_factory=lambda: {
        "warmup_steps": 500,
        "kahan_sum": True
    })

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    micro_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
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
    validation_batch_size: int = 4
    validation_scheduler: str = "ddim"
    validation_cfg_scale: float = 7.5
    max_train_steps: Optional[int] = None
    lr_scheduler: str = "linear"
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    optimizer_eps: float = 1e-8
    use_wandb: bool = True
    random_flip: bool = True
    center_crop: bool = True
    method: str = "ddpm"  # ddpm, flow_matching, consistency, dpm
    enable_gradient_clipping: bool = True
    clip_grad_value: Optional[float] = None
    clip_grad_norm: Optional[float] = 1.0
    enable_amp: bool = True
    amp_dtype: str = "float16"
    prediction_type: str = "v_prediction"
    zero_terminal_snr: bool = True
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
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""
    num_gpu_workers: int = 1
    num_cpu_workers: int = 4
    num_io_workers: int = 2
    prefetch_factor: int = 2
    use_pinned_memory: bool = True
    stream_timeout: float = 10.0
    enable_dali: bool = True
    dali_device_id: int = 0
    dali_prefetch_queue_depth: int = 2
    dali_output_dtype: str = "float32"
    enable_async_loading: bool = True

@dataclass
class Config:
    """Complete training configuration."""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tag_weighting: TagWeightingConfig = field(default_factory=TagWeightingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)

    def __iter__(self):
        """Make Config iterable to access its sections."""
        for field in [
            self.global_config,
            self.model,
            self.training,
            self.data,
            self.tag_weighting,
            self.preprocessing,
            self.transforms
        ]:
            yield field

    def __post_init__(self):
        from .utils.paths import convert_windows_path

        # Ensure output paths exist
        output_dir = convert_windows_path(self.global_config.output_dir, make_absolute=True)
        cache_dir = convert_windows_path(self.global_config.cache.cache_dir, make_absolute=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Validate image sizes
        try:
            max_size = tuple(int(x) for x in self.global_config.image.max_size)
            min_size = tuple(int(x) for x in self.global_config.image.min_size)
            self.global_config.image.max_size = max_size
            self.global_config.image.min_size = min_size
            if len(max_size) != 2 or len(min_size) != 2:
                raise ValueError("max_size and min_size must each have 2 values.")
            max_h, max_w = max_size
            min_h, min_w = min_size
            if max_h < min_h or max_w < min_w:
                raise ValueError(
                    f"max_size ({max_size}) must be >= min_size ({min_size}) in both dimensions."
                )
        except Exception as e:
            logger.error("Image size validation failed", exc_info=True)
            raise

        # Validate basic training hyperparams
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.training.batch_size < self.training.gradient_accumulation_steps:
            raise ValueError("batch_size must be >= gradient_accumulation_steps.")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        try:
            global_config_dict = config_dict.get("global_config", {})
            global_config = GlobalConfig(
                image=GlobalConfig.ImageConfig(**global_config_dict.get("image", {})),
                cache=GlobalConfig.CacheConfig(**global_config_dict.get("cache", {})),
                seed=global_config_dict.get("seed"),
                output_dir=global_config_dict.get("output_dir", "outputs"),
            )

            model_config = ModelConfig(**config_dict.get("model", {}))

            training_dict = config_dict.get("training", {})
            training_config = TrainingConfig(
                memory=MemoryConfig(**training_dict.get("memory", {})),
                flow_matching=FlowMatchingConfig(**training_dict.get("flow_matching", {})),
                **{
                    k: v
                    for k, v in training_dict.items()
                    if k not in ["memory", "flow_matching"]
                },
            )

            data_config = DataConfig(**config_dict.get("data", {}))
            tag_weighting_config = TagWeightingConfig(
                **config_dict.get("tag_weighting", {})
            )

            # If missing root-level fields, they get defaulted by the dataclasses
            return cls(
                global_config=global_config,
                model=model_config,
                training=training_config,
                data=data_config,
                tag_weighting=tag_weighting_config,
            )

        except Exception as e:
            logger.error(
                f"Failed to parse config from YAML ({path}): {str(e)}", exc_info=True
            )
            raise
