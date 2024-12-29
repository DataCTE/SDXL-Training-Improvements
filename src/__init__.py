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
    WandbLogger
)

from .data import (
    Config,
    AspectBucketDataset,
    create_dataset,
    PreprocessingPipeline,
    CacheManager,
    TagWeighter,
    create_tag_weighter_with_index
)

from .models import (
    BaseModel,
    BaseModelEmbedding,
    ModelType,
    StableDiffusionXL,
    TimestepBiasStrategy,
    CLIPEncoder,
    VAEEncoder
)

from .training import (
    BaseTrainer,
    SDXLTrainer,
    DDPMScheduler,
    TrainingMethod,
    DDPMTrainer,
    FlowMatchingTrainer,
    configure_noise_scheduler,
    get_karras_sigmas,
    get_sigmas,
    get_scheduler_parameters,
    get_add_time_ids,
    create_trainer
)

__version__ = "0.1.0"

from typing import List

__all__: List[str] = [
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
    "WandbLogger",

    # Data
    "Config",
    "AspectBucketDataset",
    "create_dataset",
    "PreprocessingPipeline",
    "CacheManager",
    "TagWeighter",
    "create_tag_weighter_with_index",

    # Models
    "BaseModel",
    "BaseModelEmbedding",
    "ModelType",
    "StableDiffusionXL",
    "TimestepBiasStrategy",
    "CLIPEncoder",
    "VAEEncoder",

    # Training components
    "BaseTrainer",
    "SDXLTrainer",
    "DDPMScheduler",
    "TrainingMethod",
    "DDPMTrainer",
    "FlowMatchingTrainer",
    "configure_noise_scheduler",
    "get_karras_sigmas",
    "get_sigmas",
    "get_scheduler_parameters",
    "get_add_time_ids",
    "create_trainer"
]
