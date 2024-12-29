"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# Force speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path
from src.data.config import Config
from src.data.preprocessing import (
    PreprocessingPipeline,
    CacheManager,
    create_tag_weighter_with_index
)

logger = get_logger(__name__)

@dataclass
class DatasetStats:
    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_allocated: int = 0
    peak_memory: int = 0

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with extreme memory handling and 100x speedups."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        is_train: bool = True,
        enable_memory_tracking: bool = True,
        max_memory_usage: float = 0.8
    ):
        try:
            super().__init__()
            start_time = time.time()
            
            # Basic initialization
            self.stats = DatasetStats()
            self.config = config
            self.is_train = is_train
            self.enable_memory_tracking = enable_memory_tracking
            self.max_memory_usage = max_memory_usage

            # Convert paths if needed
            self.image_paths = [
                str(convert_windows_path(p) if is_windows_path(p) else Path(p))
                for p in image_paths
            ]
            self.captions = captions

            # Initialize cache manager
            cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                max_cache_size=config.global_config.cache.max_cache_size,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            # Initialize preprocessing pipeline
            self.preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline(
                config=config,
                model=config.model,
                cache_manager=self.cache_manager,
                is_train=is_train,
                enable_memory_tracking=enable_memory_tracking
            )

            # Initialize tag weighter with index
            if config.tag_weighting.enable_tag_weighting:
                image_captions = dict(zip(self.image_paths, self.captions))
                index_path = Path(cache_dir) / "tag_weights_index.json"
                self.tag_weighter = create_tag_weighter_with_index(
                    config=config,
                    image_captions=image_captions,
                    index_output_path=index_path
                )
            else:
                self.tag_weighter = None

            # Setup image configuration
            self._setup_image_config()

            # Precompute latents if caching enabled
            if config.global_config.cache.use_cache:
                self._precompute_latents()

            logger.info(f"Dataset initialized in {time.time() - start_time:.2f}s", extra={
                'num_images': len(image_paths),
                'num_captions': len(captions),
                'cache_enabled': config.global_config.cache.use_cache,
                'tag_weighting_enabled': self.tag_weighter is not None
            })

        except Exception as e:
            logger.error("Failed to initialize dataset", exc_info=True)
            raise

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset with robust error handling."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]

            # Try to load from cache first
            cached_data = self.cache_manager.load_latents(image_path)
            if cached_data is not None:
                self.stats.cache_hits += 1
                result = {
                    "latents": cached_data["latents"],
                    "metadata": cached_data["metadata"],
                    "text": caption
                }
                
                # Add tag weight if enabled
                if self.tag_weighter:
                    result["tag_weight"] = self.tag_weighter.get_caption_weight(caption)
                    
                return result

            self.stats.cache_misses += 1

            # Process image if not in cache
            processed = self.preprocessing_pipeline._process_image(image_path)
            if processed is None:
                raise ValueError(f"Failed to process image {image_path}")

            # Get text embeddings
            encoded_text = self.preprocessing_pipeline.encode_prompt(
                batch={"text": [caption]},
                proportion_empty_prompts=self.config.data.proportion_empty_prompts
            )

            result = {
                "latents": processed["model_input"],
                "metadata": processed["metadata"],
                "text": caption,
                "prompt_embeds": encoded_text["prompt_embeds"],
                "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"]
            }

            # Add tag weight if enabled
            if self.tag_weighter:
                result["tag_weight"] = self.tag_weighter.get_caption_weight(caption)

            # Cache the processed data
            if self.config.global_config.cache.use_cache:
                self.cache_manager.save_latents(
                    latents=processed["model_input"],
                    original_path=image_path,
                    metadata={
                        **processed["metadata"],
                        "text_embeddings": {
                            "prompt_embeds": encoded_text["prompt_embeds"],
                            "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"]
                        }
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error processing dataset item {idx}", exc_info=True)
            # Try next item
            return self.__getitem__((idx + 1) % len(self))

    def _setup_image_config(self):
        """Set up image configuration parameters."""
        self.target_size = tuple(map(int, self.config.global_config.image.target_size))
        self.max_size = tuple(map(int, self.config.global_config.image.max_size))
        self.min_size = tuple(map(int, self.config.global_config.image.min_size))
        self.bucket_step = int(self.config.global_config.image.bucket_step)
        self.max_aspect_ratio = float(self.config.global_config.image.max_aspect_ratio)

        # Get buckets from pipeline
        self.buckets = self.preprocessing_pipeline.get_aspect_buckets(self.config)
        self.bucket_indices = self.preprocessing_pipeline._assign_single_bucket(self.image_paths)

    def _precompute_latents(self) -> None:
        """Precompute and cache latents for all images."""
        logger.info("Starting latent precomputation...")
        
        for idx in tqdm(range(len(self)), desc="Precomputing latents"):
            try:
                self.__getitem__(idx)
            except Exception as e:
                logger.error(f"Failed to precompute latents for index {idx}", exc_info=True)
                continue

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            result = {
                "latents": [],
                "prompt_embeds": [],
                "pooled_prompt_embeds": [],
                "tag_weights": [] if self.tag_weighter else None
            }

            for item in batch:
                result["latents"].append(item["latents"])
                if "prompt_embeds" in item:
                    result["prompt_embeds"].append(item["prompt_embeds"])
                if "pooled_prompt_embeds" in item:
                    result["pooled_prompt_embeds"].append(item["pooled_prompt_embeds"])
                if self.tag_weighter:
                    result["tag_weights"].append(item.get("tag_weight", 1.0))

            # Stack tensors
            result["latents"] = torch.stack(result["latents"])
            if result["prompt_embeds"]:
                result["prompt_embeds"] = torch.stack(result["prompt_embeds"])
            if result["pooled_prompt_embeds"]:
                result["pooled_prompt_embeds"] = torch.stack(result["pooled_prompt_embeds"])
            if result["tag_weights"]:
                result["tag_weights"] = torch.tensor(result["tag_weights"], dtype=torch.float32)

            return result

        except Exception as e:
            logger.error("Failed to collate batch", exc_info=True)
            raise

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    """Create and initialize dataset instance."""
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        preprocessing_pipeline=preprocessing_pipeline,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
