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
    CLIPEncoder
)

from .training import (
    SDXLTrainer,
    DDPMScheduler,
    TrainingMethod,
    DDPMTrainer,
    FlowMatchingTrainer,
    configure_noise_scheduler,
    get_karras_sigmas,
    get_sigmas,
    get_scheduler_parameters,
    get_add_time_ids
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
    "CLIPEncoder",

    # Training components
    "SDXLTrainer",
    "DDPMScheduler",
    "TrainingMethod",
    "DDPMTrainer",
    "FlowMatchingTrainer", 
    "configure_noise_scheduler",
    "get_karras_sigmas",
    "get_sigmas",
    "get_scheduler_parameters",
    "get_add_time_ids"
]
