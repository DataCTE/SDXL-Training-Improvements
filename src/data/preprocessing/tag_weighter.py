"""Tag weighting system for SDXL training with JSON index support."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any
import numpy as np
import torch
import time
from tqdm import tqdm

if TYPE_CHECKING:
    from src.data.config import Config

from src.core.logging import get_logger, LogConfig
from src.data.utils.paths import convert_windows_path
from src.models.sdxl import StableDiffusionXL
from src.models.encoders import CLIPEncoder
from src.data.preprocessing.exceptions import TagProcessingError

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
        
        # Initialize tag types and embeddings
        self.tag_types = {
            "subject": ["person", "object", "animal", "vehicle", "building", "landscape"],
            "style": ["painting", "anime", "drawing", "photograph", "digital art", "sketch", "watercolor"],
            "quality": ["high quality", "detailed", "sharp", "professional", "masterpiece"],
            "technical": ["lighting", "composition", "focus", "exposure", "angle", "depth"]
        }
        
        # Initialize counters with proper defaults
        self.tag_counts = defaultdict(DefaultDict(default_int))
        self.tag_weights = defaultdict(DefaultDict(lambda: self.default_weight))
        self.total_samples = 0
        
        # Cache reference
        self.cache_manager = config.cache_manager if hasattr(config, 'cache_manager') else None

    def verify_cache_validity(self) -> bool:
        """Verify that loaded cache data is valid and complete."""
        try:
            # Check basic structure
            if not self.tag_counts or not self.tag_weights:
                return False
                
            # Verify all tag types are present
            for tag_type in self.tag_types:
                if tag_type not in self.tag_counts or tag_type not in self.tag_weights:
                    return False
                    
            # Verify weight ranges
            for tag_type, weights in self.tag_weights.items():
                if not weights:  # Skip empty categories
                    continue
                min_weight = min(weights.values())
                max_weight = max(weights.values())
                if not (self.min_weight <= min_weight <= max_weight <= self.max_weight):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False

    def _initialize_category_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        """Initialize category embeddings using CLIP."""
        if not self.clip_encoder or not self.tokenizer:
            return None
        
        embeddings = {}
        with torch.no_grad():
            for category, terms in self.tag_types.items():
                # Encode category terms
                text_embeddings = CLIPEncoder.encode_prompt(
                    batch={"text": terms},
                    text_encoders=[self.clip_encoder.text_encoder],
                    tokenizers=[self.tokenizer],
                    is_train=False
                )
                # Use pooled embeddings for category representation
                embeddings[category] = torch.mean(
                    text_embeddings["pooled_prompt_embeds"], 
                    dim=0
                )
        
        return embeddings

    def _get_semantic_category(self, phrase: str) -> str:
        """Determine semantic category using CLIP similarity."""
        if not hasattr(self, 'category_embeddings'):
            self.category_embeddings = self._initialize_category_embeddings()
        
        if not self.clip_encoder or not self.category_embeddings:
            return "meta"
        
        with torch.no_grad():
            # Get phrase embedding
            phrase_embedding = CLIPEncoder.encode_prompt(
                batch={"text": [phrase]},
                text_encoders=[self.clip_encoder.text_encoder],
                tokenizers=[self.tokenizer],
                is_train=False
            )["pooled_prompt_embeds"]
            
            # Compare with category embeddings
            similarities = {
                category: torch.cosine_similarity(
                    phrase_embedding, 
                    cat_embedding.unsqueeze(0)
                ).item()
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
                if len(tag) > 100:  # Skip overly long tags
                    continue
                    
                if ':' in tag:
                    parts = tag.lower().split(':')
                    category = parts[0]
                    if category in categorized:
                        original_tag = ':'.join(tag.split(':')[1:])
                        categorized[category].append(original_tag)
                        continue
                
                # Use semantic categorization for natural language
                category = self._get_semantic_category(tag)
                categorized[category].append(tag)
                    
            return categorized
            
        except Exception as e:
            logger.error(f"Tag extraction failed: {e}")
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
        
        # Process in larger batches for efficiency
        batch_size = 5000
        for i in tqdm(range(0, len(captions), batch_size), desc="Processing captions"):
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
        
        # Save to cache if enabled
        if self.config.tag_weighting.use_cache:
            self.save_to_cache()

    def _compute_weights(self) -> None:
        """Compute tag weights based on in-class frequency with proper scaling."""
        for tag_type in self.tag_counts:
            type_counts = self.tag_counts[tag_type]
            total_type_count = sum(type_counts.values())
            
            if total_type_count == 0:
                continue
                
            for tag, count in type_counts.items():
                # Calculate frequency within this tag type class
                in_class_frequency = count / total_type_count
                
                # Apply inverse in-class frequency with smoothing
                raw_weight = 1.0 / (in_class_frequency + self.smoothing_factor)
                
                # Scale to desired range while preserving relative weights
                max_possible_weight = 1.0 / self.smoothing_factor
                normalized_weight = (raw_weight - 1.0) / (max_possible_weight - 1.0)
                scaled_weight = (
                    self.min_weight +
                    (normalized_weight * (self.max_weight - self.min_weight))
                )
                
                # Save the computed weight
                self.tag_weights[tag_type][tag] = float(
                    min(max(scaled_weight, self.min_weight), self.max_weight)
                )

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

    def _load_cache(self) -> bool:
        """Load tag statistics from cache with validation."""
        try:
            if not self.cache_manager:
                return False
            
            index_data = self.cache_manager.load_tag_index()
            if not index_data:
                return False
            
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
            
            # Verify loaded data
            return self.verify_cache_validity()
            
        except Exception as e:
            logger.error(f"Failed to load tag cache: {e}")
            return False

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

    def _prepare_index_data(self, image_tags: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare tag index data for saving."""
        return {
            "version": "1.0",
            "updated_at": time.time(),
            "metadata": {
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
                }
            },
            "statistics": {
                "tag_counts": {
                    tag_type: dict(counts)
                    for tag_type, counts in self.tag_counts.items()
                },
                "tag_weights": {
                    tag_type: dict(weights)
                    for tag_type, weights in self.tag_weights.items()
                }
            },
            "images": image_tags
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

    def process_dataset_tags(self, captions: List[str]) -> Dict[str, Any]:
        """Process all dataset captions and return tag information."""
        processed_tags = {}
        
        with logger.start_progress(
            total=len(captions),
            desc="Processing tags"
        ) as progress:
            for caption in captions:
                tags = self._extract_tags(caption)
                weighted_tags = {
                    tag_type: [
                        {"tag": tag, "weight": self.tag_weights[tag_type][tag]}
                        for tag in tags[tag_type]
                    ]
                    for tag_type in tags
                }
                
                caption_weight = self.get_caption_weight(caption)
                processed_tags[caption] = {
                    "tags": weighted_tags,
                    "weight": caption_weight
                }
                
                # Update progress with statistics
                progress.update(1, {
                    "unique_tags": {
                        t: len(self.tag_counts[t]) 
                        for t in self.tag_types
                    }
                })
                
        return processed_tags

    def get_tag_metadata(self) -> Dict[str, Any]:
        """Get current tag statistics and metadata.
        
        Returns:
            Dict containing tag statistics and metadata
        """
        return {
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
            "tag_weights": {
                tag_type: dict(weights)
                for tag_type, weights in self.tag_weights.items()
            }
        }

    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get detailed tag statistics.
        
        Returns:
            Dict containing detailed tag statistics
        """
        return {
            "tag_counts": {
                tag_type: dict(counts)
                for tag_type, counts in self.tag_counts.items()
            },
            "tag_weights": {
                tag_type: dict(weights)
                for tag_type, weights in self.tag_weights.items()
            },
            "total_samples": self.total_samples
        }

    def get_tag_info(self, caption: str) -> Dict[str, Any]:
        """Get tag information for a single caption.
        
        Args:
            caption: Image caption to process
            
        Returns:
            Dict containing tag information and weights
        """
        try:
            tags = self._extract_tags(caption)
            
            tag_info = {
                "tags": {
                    tag_type: [
                        {
                            "tag": tag,
                            "weight": self.tag_weights[tag_type][tag]
                        }
                        for tag in tags[tag_type]
                    ]
                    for tag_type in tags
                },
                "weight": self.get_caption_weight(caption)
            }
            
            return tag_info
            
        except Exception as e:
            logger.error(f"Failed to get tag info for caption: {e}")
            return {"tags": {}, "weight": self.default_weight}

    def save_to_cache(self) -> bool:
        """Save current tag weights and statistics to cache.
        
        Returns:
            bool: True if save was successful
        """
        try:
            if not self.cache_manager:
                return False
            
            # Prepare tag data
            tag_data = {
                "version": "1.0",
                "updated_at": time.time(),
                "metadata": {
                    "config": {
                        "default_weight": self.default_weight,
                        "min_weight": self.min_weight,
                        "max_weight": self.max_weight,
                        "smoothing_factor": self.smoothing_factor
                    }
                },
                "statistics": self.get_tag_statistics()
            }
            
            # Save through cache manager
            self.cache_manager.save_tag_index(tag_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tag data to cache: {e}")
            return False

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

    def initialize_tag_system(self) -> bool:
        """Initialize tag system with enhanced validation."""
        try:
            if not self.cache_manager:
                return False
            
            if self.config.tag_weighting.use_cache:
                tag_data = self.cache_manager.load_tag_index()
                if tag_data and self._validate_tag_data(tag_data):
                    return self._load_cache()
                
            return self._initialize_fresh()
            
        except Exception as e:
            raise TagProcessingError("Tag system initialization failed", 
                context={"error": str(e), "cache_enabled": self.config.tag_weighting.use_cache})

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
