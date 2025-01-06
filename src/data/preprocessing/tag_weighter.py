"""Tag weighting system for SDXL training with JSON index support."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any
import numpy as np
import torch
import time

if TYPE_CHECKING:
    from src.data.config import Config

from src.core.logging import get_logger, LogConfig
from src.data.utils.paths import convert_windows_path
from src.models.sdxl import StableDiffusionXL

logger = get_logger(__name__)

def default_int():
    return 0

class DefaultDict:
    def __init__(self, default_factory):
        self.default_factory = default_factory
        
    def __call__(self):
        return defaultdict(self.default_factory)

class TagWeighter:
    def __init__(
        self,
        config: "Config",  # type: ignore
        model: Optional["StableDiffusionXL"] = None  # Add model parameter
    ):
        """Initialize tag weighting system."""
        self.config = config
        
        # Core settings
        self.default_weight = config.tag_weighting.default_weight
        self.min_weight = config.tag_weighting.min_weight
        self.max_weight = config.tag_weighting.max_weight
        self.smoothing_factor = config.tag_weighting.smoothing_factor
        
        # Use SDXL's CLIP encoders
        self.clip_encoder = model.clip_encoder_1 if model else None
        self.tokenizer = model.tokenizer_1 if model else None
        
        # Category embeddings using CLIP
        self.category_embeddings = self._initialize_category_embeddings([
            ("subject", ["person", "object", "animal", "vehicle", "building", "landscape"]),
            ("style", ["painting", "anime", "drawing", "photograph", "digital art", "sketch", "watercolor"]),
            ("quality", ["high quality", "detailed", "sharp", "professional", "masterpiece"]),
            ("technical", ["lighting", "composition", "focus", "exposure", "angle", "depth"])
        ])
        
        # Initialize counters
        self.tag_counts = defaultdict(DefaultDict(default_int))
        self.tag_weights = defaultdict(DefaultDict(lambda: self.default_weight))
        self.total_samples = 0
        
        # Cache reference
        self.cache_manager = config.cache_manager if hasattr(config, 'cache_manager') else None

    def _initialize_category_embeddings(self, categories):
        """Initialize category embeddings using SDXL's CLIP."""
        if not self.clip_encoder:
            return None
        
        embeddings = {}
        with torch.no_grad():
            for category, terms in categories:
                # Use CLIP to get embeddings
                text_embeddings = self.clip_encoder.encode_text(terms)
                embeddings[category] = torch.mean(text_embeddings, dim=0)
        
        return embeddings

    def _get_semantic_category(self, phrase: str) -> str:
        """Determine category using CLIP similarity."""
        if not self.clip_encoder:
            return "meta"
        
        with torch.no_grad():
            # Get phrase embedding using CLIP
            phrase_embedding = self.clip_encoder.encode_text([phrase])
            
            # Compare with category embeddings
            similarities = {
                category: torch.cosine_similarity(
                    phrase_embedding, 
                    cat_embedding.unsqueeze(0)
                )
                for category, cat_embedding in self.category_embeddings.items()
            }
            
            return max(similarities.items(), key=lambda x: x[1])[0]

    def _extract_tags(self, caption: str) -> Dict[str, List[str]]:
        """Extract and categorize tags from caption."""
        try:
            categorized = {
                "subject": [],
                "style": [],
                "quality": [],
                "technical": [],
                "meta": []
            }
            
            # Handle explicit category:tag format first
            for tag in (t.strip() for t in caption.split(',') if t):
                if len(tag) > 100:
                    continue
                    
                if ':' in tag:
                    parts = tag.lower().split(':')
                    category = parts[0]
                    if category in categorized:
                        original_tag = ':'.join(tag.split(':')[1:])
                        categorized[category].append(original_tag)
                        continue
                
                # Semantic categorization for natural language
                category = self._get_semantic_category(tag)
                categorized[category].append(tag)
                    
            return categorized
            
        except Exception as e:
            return {
                "subject": [], "style": [], "quality": [], 
                "technical": [], "meta": []
            }

    def update_statistics(self, captions: List[str]) -> None:
        """Update tag statistics from captions."""
        # Pre-allocate counters
        tag_counts = {
            "subject": defaultdict(int),
            "style": defaultdict(int),
            "quality": defaultdict(int),
            "technical": defaultdict(int),
            "meta": defaultdict(int)
        }
        
        # Process in larger batches
        batch_size = 5000
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i + batch_size]
            self.total_samples += len(batch)
            
            # Batch process tags
            for caption in batch:
                for tag_type, tags in self._extract_tags(caption).items():
                    for tag in tags:
                        tag_counts[tag_type][tag] += 1
        
        # Bulk update counts
        for tag_type, counts in tag_counts.items():
            self.tag_counts[tag_type].update(counts)
        
        self._compute_weights()
        
        if self.config.tag_weighting.use_cache:
            self._save_cache()

    def _compute_weights(self) -> None:
        """Compute tag weights based on in-class frequency with proper scaling."""
        for tag_type in self.tag_counts:
            type_counts = self.tag_counts[tag_type]
            total_type_count = sum(type_counts.values())
            
            if total_type_count == 0:
                continue
                
            for tag, count in type_counts.items():
                # Calculate frequency within this tag type class
                in_class_frequency = count / total_type_count  # Changed from total_samples
                
                # Apply inverse in-class frequency with smoothing
                raw_weight = 1.0 / (in_class_frequency + self.smoothing_factor)
                
                # Scale to desired range while preserving relative in-class weights
                max_possible_weight = 1.0 / self.smoothing_factor
                normalized_weight = (raw_weight - 1.0) / (max_possible_weight - 1.0)
                scaled_weight = (
                    self.min_weight +
                    (normalized_weight * (self.max_weight - self.min_weight))
                )
                
                # Save the computed weight for this tag
                self.tag_weights[tag_type][tag] = float(min(max(scaled_weight, self.min_weight), self.max_weight))

            # Save weights if caching is enabled
            if self.config.tag_weighting.use_cache:
                self._save_cache()

    def get_caption_weight(self, caption: str) -> float:
        """Get combined weight for a caption."""
        categorized_tags = self._extract_tags(caption)
        weights = []
        
        for tag_type, tags in categorized_tags.items():
            type_weights = [self.tag_weights[tag_type][tag] for tag in tags]
            if type_weights:
                weights.append(np.mean(type_weights))
        
        if not weights:
            return self.default_weight
            
        return float(np.exp(np.mean(np.log(weights))))

    def get_caption_weight_details(self, caption: str) -> Dict[str, Any]:
        """Get detailed weights for all tags in a caption with enhanced metadata.
        
        Args:
            caption: Input caption to analyze
            
        Returns:
            Dict containing:
                - total_weight: float
                - tags: Dict[str, List[Dict]] with categorized tag info
                - metadata: Dict with additional tag statistics
        """
        categorized_tags = self._extract_tags(caption)
        tag_details = {
            "total_weight": self.default_weight,
            "tags": defaultdict(list),
            "metadata": {
                "tag_counts": {},
                "category_weights": {},
                "timestamp": time.time()
            }
        }
        
        weights = []
        
        # Process each tag category
        for tag_type, tags in categorized_tags.items():
            category_weights = []
            for tag in tags:
                weight = self.tag_weights[tag_type][tag]
                frequency = self.tag_counts[tag_type][tag] / self.total_samples if self.total_samples > 0 else 0
                
                tag_details["tags"][tag_type].append({
                    "tag": tag,
                    "weight": float(weight),
                    "frequency": float(frequency),
                    "count": int(self.tag_counts[tag_type][tag])
                })
                category_weights.append(weight)
            
            if category_weights:
                weights.append(np.mean(category_weights))
                tag_details["metadata"]["category_weights"][tag_type] = float(np.mean(category_weights))
        
        # Calculate total caption weight using geometric mean
        if weights:
            tag_details["total_weight"] = float(np.exp(np.mean(np.log(weights))))
        
        # Add tag count statistics
        tag_details["metadata"]["tag_counts"] = {
            tag_type: len(tags) for tag_type, tags in categorized_tags.items()
        }
        
        return dict(tag_details)

    def get_batch_weights(self, captions: List[str]) -> torch.Tensor:
        """Get weights for a batch of captions."""
        weights = [self.get_caption_weight(caption) for caption in captions]
        return torch.tensor(weights, dtype=torch.float32)

    def _save_cache(self) -> None:
        """Save tag statistics to cache."""
        if not hasattr(self.config, 'cache_manager'):
            return
        
        index_data = self._prepare_index_data({})  # Empty images dict for statistics only
        self.config.cache_manager.save_tag_index(index_data)

    def _load_cache(self) -> None:
        """Load tag statistics from cache."""
        if not hasattr(self.config, 'cache_manager'):
            return
        
        index_data = self.config.cache_manager.load_tag_index()
        if not index_data:
            return
        
        # Load statistics from index
        if "statistics" in index_data:
            stats = index_data["statistics"]
            if "tag_counts" in stats:
                for tag_type, counts in stats["tag_counts"].items():
                    self.tag_counts[tag_type].update(counts)
            
            if "tag_weights" in stats:
                for tag_type, weights in stats["tag_weights"].items():
                    self.tag_weights[tag_type].update(weights)
        
        # Load metadata
        if "metadata" in index_data:
            self.total_samples = index_data["metadata"].get("total_samples", 0)

    def get_tag_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics about tags and weights."""
        stats = {}
        
        for tag_type in self.tag_counts:
            type_stats = {
                "total_tags": len(self.tag_counts[tag_type]),
                "total_occurrences": sum(self.tag_counts[tag_type].values()),
                "most_common": sorted(
                    self.tag_counts[tag_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
            
            # Add weight range only if we have weights
            if self.tag_weights[tag_type]:
                type_stats["weight_range"] = (
                    min(self.tag_weights[tag_type].values()),
                    max(self.tag_weights[tag_type].values())
                )
            else:
                type_stats["weight_range"] = (self.default_weight, self.default_weight)
            
            stats[tag_type] = type_stats
            
        return stats

    def save_to_index(self, output_path: Path, image_tags: Dict[str, Dict[str, any]]) -> None:
        """Save tag weights and statistics to index using cache manager."""
        if not hasattr(self.config, 'cache_manager'):
            return
        
        index_data = self._prepare_index_data(image_tags)
        self.config.cache_manager.save_tag_index(index_data)

    def _prepare_index_data(self, image_tags: Dict[str, Dict[str, any]]) -> Dict[str, Any]:
        """Prepare index data structure for global metadata and per-image tags."""
        return {
            "version": "1.0",
            "metadata": {
                "total_samples": self.total_samples,
                "default_weight": self.default_weight,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "smoothing_factor": self.smoothing_factor,
                "tag_types": self.tag_types,
                "created_at": time.time(),
                "statistics": {
                    "tag_counts": {k: dict(v) for k, v in self.tag_counts.items()},
                    "tag_weights": {k: dict(v) for k, v in self.tag_weights.items()},
                    "type_statistics": self.get_tag_statistics()
                }
            },
            "images": {
                str(image_path): {
                    "caption": image_data["caption"],
                    "tags": {
                        category: [
                            {
                                "tag": tag_info["tag"],
                                "weight": float(tag_info["weight"]),
                                "frequency": float(self.tag_counts[category][tag_info["tag"]] / self.total_samples)
                            }
                            for tag_info in tags_list
                        ]
                        for category, tags_list in image_data["weight_details"]["tags"].items()
                    }
                }
                for image_path, image_data in image_tags.items()
            }
        }

    def _clean_numeric_values(self, obj: Any) -> Any:
        """Clean numeric values for JSON serialization."""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._clean_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_numeric_values(x) for x in obj]
        return obj

    def process_dataset_tags(self, image_captions: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Process all dataset tags and return weighted tags per image."""
        image_tags = {}
        
        for image_path, caption in image_captions.items():
            categorized_tags = self._extract_tags(caption)
            
            # Store tags with their individual weights
            image_tags[image_path] = {
                "tags": {
                    category: [
                        {"tag": tag, "weight": self.tag_weights[category][tag]}
                        for tag in tags
                    ]
                    for category, tags in categorized_tags.items()
                }
            }
            
        return image_tags

    def get_tag_metadata(self, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata about tag weighting for caching purposes."""
        metadata = {
            "config": {
                "default_weight": self.default_weight,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "smoothing_factor": self.smoothing_factor
            },
            "statistics": {
                "total_samples": self.total_samples,
                "tag_type_counts": {
                    tag_type: sum(counts.values())
                    for tag_type, counts in self.tag_counts.items()
                },
                "unique_tags": {
                    tag_type: len(counts)
                    for tag_type, counts in self.tag_counts.items()
                }
            },
            "tag_types": self.tag_types,
            "timestamp": time.time()
        }
        
        if cache_key:
            metadata["cache_key"] = cache_key
            
        return metadata
    
    def __getstate__(self):
        """Customize pickling behavior."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Customize unpickling behavior."""
        self.__dict__.update(state)

    def get_image_tag_weights(self, image_path: str) -> Dict[str, Any]:
        """Load tag weights for an image from cache."""
        if not self.cache_manager:
            return {
                "tags": {
                    category: [] for category in self.tag_types.keys()
                }
            }
            
        # Load from cache
        tag_index = self.cache_manager.load_tag_index()
        if not tag_index or "images" not in tag_index:
            return {
                "tags": {
                    category: [] for category in self.tag_types.keys()
                }
            }
            
        # Get image-specific tags with weights
        image_tags = tag_index["images"].get(image_path, {})
        return {
            "tags": {
                category: [
                    {"tag": tag, "weight": self.tag_weights[category][tag]}
                    for tag in tags
                ]
                for category, tags in image_tags.get("tags", {}).items()
            }
        }

def create_tag_weighter(
    config: "Config",  # type: ignore
    captions: List[str],
    cache_path: Optional[Path] = None
) -> TagWeighter:
    """Create and initialize a tag weighter."""
    weighter = TagWeighter(config, cache_path)
    
    if not (config.tag_weighting.use_cache and weighter.cache_path.exists()):
        logger.info("Computing tag statistics...")
        weighter.update_statistics(captions)
        
    stats = weighter.get_tag_statistics()
    logger.info("Tag statistics:")
    for tag_type, type_stats in stats.items():
        logger.info(f"\n{tag_type}:")
        logger.info(f"Total unique tags: {type_stats['total_tags']}")
        logger.info(f"Total occurrences: {type_stats['total_occurrences']}")
        logger.info(f"Weight range: {type_stats['weight_range']}")
        
    return weighter

def create_tag_weighter_with_index(
    config: "Config",
    image_captions: Dict[str, str],
    model: Optional["StableDiffusionXL"] = None  # Add model parameter
) -> TagWeighter:
    """Create and initialize tag weighter with index."""
    weighter = TagWeighter(config, model=model)  # Pass model to TagWeighter
    
    logger.info("Processing captions and updating tag statistics...")
    weighter.update_statistics(list(image_captions.values()))
    
    logger.info("Creating detailed tag index...")
    image_tags = weighter.process_dataset_tags(image_captions)
    
    # Save to cache manager's tag directory
    logger.info("Saving tag index to cache")
    weighter.save_to_index(None, image_tags)
    
    return weighter

def preprocess_dataset_tags(
    config: "Config",
    image_paths: List[str],
    captions: List[str]
) -> Optional[TagWeighter]:
    """Preprocess all dataset tags before training."""
    if not config.tag_weighting.enable_tag_weighting:
        return None
        
    logger.info("Starting tag preprocessing...")
    
    # Get tag index path from cache manager
    if not hasattr(config, 'cache_manager'):
        logger.warning("Cache manager not available for tag preprocessing")
        return None
        
    # Create and initialize tag weighter
    image_captions = dict(zip(image_paths, captions))
    
    logger.info("Processing tags and creating index...")
    weighter = create_tag_weighter_with_index(
        config=config,
        image_captions=image_captions
    )
    
    logger.info("Tag preprocessing complete")
    return weighter
