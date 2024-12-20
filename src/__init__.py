"""SDXL training package."""
from .core import (
    DataType,
    ModelWeightDtypes,
    LayerOffloader,
    LayerOffloadConfig,
    setup_memory_optimizations,
    verify_memory_optimizations,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    reduce_dict,
    setup_logging,
    cleanup_logging,
    WandbLogger
)

from .data import (
    Config,
    AspectBucketDataset,
    create_dataset,
    LatentPreprocessor,
    CacheManager,
    TagWeighter,
    create_tag_weighter,
    PreprocessingPipeline
)

from .models import (
    BaseModel,
    BaseModelEmbedding,
    ModelType,
    StableDiffusionXLModel,
    StableDiffusionXLModelEmbedding,
    LoRAModuleWrapper,
    AdditionalEmbeddingWrapper,
    encode_clip
)

from .training import (
    SDXLTrainer,
    DDPMScheduler,
    configure_noise_scheduler,
    get_karras_scalings,
    get_sigmas,
    get_scheduler_parameters,
    generate_noise,
    get_add_time_ids,
    log_metrics,
    sample_logit_normal,
    optimal_transport_path,
    compute_flow_matching_loss
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "DataType",
    "ModelWeightDtypes",
    "LayerOffloader",
    "LayerOffloadConfig",
    "setup_memory_optimizations",
    "verify_memory_optimizations",
    "setup_distributed",
    "cleanup_distributed", 
    "is_main_process",
    "get_world_size",
    "reduce_dict",
    "setup_logging",
    "cleanup_logging",
    "WandbLogger",

    # Data
    "Config",
    "AspectBucketDataset",
    "create_dataset",
    "LatentPreprocessor",
    "CacheManager", 
    "TagWeighter",
    "create_tag_weighter",
    "PreprocessingPipeline",

    # Models
    "BaseModel",
    "BaseModelEmbedding",
    "ModelType",
    "StableDiffusionXLModel",
    "StableDiffusionXLModelEmbedding",
    "LoRAModuleWrapper",
    "AdditionalEmbeddingWrapper",
    "encode_clip",

    # Training
    "SDXLTrainer",
    "DDPMScheduler",
    "configure_noise_scheduler",
    "get_karras_scalings", 
    "get_sigmas",
    "get_scheduler_parameters",
    "generate_noise",
    "get_add_time_ids",
    "log_metrics",
    "sample_logit_normal",
    "optimal_transport_path",
    "compute_flow_matching_loss"
]
