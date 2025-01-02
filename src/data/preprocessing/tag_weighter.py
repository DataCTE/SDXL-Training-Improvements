"""Tag weighting system for SDXL training with JSON index support."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any
import numpy as np
import torch

if TYPE_CHECKING:
    from src.data.config import Config

from src.core.logging import get_logger, LogConfig
from src.data.utils.paths import convert_windows_path

logger = get_logger(__name__)

class TagWeighter:
    def __init__(
        self,
        config: "Config",  # type: ignore
        cache_path: Optional[Path] = None
    ):
        """Initialize tag weighting system."""
        self.config = config
        
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True)
        self.cache_path = Path(convert_windows_path(
            cache_path if cache_path else cache_dir / "tag_weights.json",
            make_absolute=True
        ))
        
        # Tag weighting settings
        self.default_weight = config.tag_weighting.default_weight
        self.min_weight = config.tag_weighting.min_weight
        self.max_weight = config.tag_weighting.max_weight
        self.smoothing_factor = config.tag_weighting.smoothing_factor
        
        # Initialize tag statistics
        self.tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_weights = defaultdict(lambda: defaultdict(lambda: self.default_weight))
        self.total_samples = 0
        
        # Tag type categories  
        self.tag_types = {
            "subject": ["person", "character", "object", "animal", "vehicle", "location"],
            "style": ["art_style", "artist", "medium", "genre"],
            "quality": ["quality", "rating", "aesthetic"],
            "technical": ["camera", "lighting", "composition", "color"],
            "meta": ["source", "meta", "misc"]
        }
        
        # Load cached weights if available
        if config.tag_weighting.use_cache and self.cache_path.exists():
            self._load_cache()

    def _extract_tags(self, caption: str) -> Dict[str, List[str]]:
        """Extract and categorize tags from caption."""
        try:
            # More robust tag splitting
            tags = [t.strip() for t in caption.split(',') if t.strip()]
            
            # Initialize categories
            categorized = {category: [] for category in self.tag_types.keys()}
            
            for tag in tags:
                # Skip empty or malformed tags
                if not tag or len(tag) > 100:  # Sanity check for tag length
                    continue
                    
                # Default to 'meta' category if no match found
                matched = False
                for category, keywords in self.tag_types.items():
                    if any(keyword in tag.lower() for keyword in keywords):
                        categorized[category].append(tag)
                        matched = True
                        break
                
                if not matched:
                    categorized["meta"].append(tag)
                    
            return categorized
            
        except Exception as e:
            logger.warning(f"Failed to extract tags from caption: {caption[:50]}...")
            return {category: [] for category in self.tag_types.keys()}

    def _determine_tag_type(self, tag: str) -> str:
        """Determine tag type based on keywords.
        
        Args:
            tag: Input tag to categorize
            
        Returns:
            Category name for the tag
        """
        tag = tag.lower()
        for type_name, keywords in self.tag_types.items():
            if any(keyword in tag for keyword in keywords):
                return type_name
        return "subject"

    def update_statistics(self, captions: List[str]) -> None:
        """Update tag statistics from captions."""
        for caption in captions:
            self.total_samples += 1
            categorized_tags = self._extract_tags(caption)
            
            for tag_type, tags in categorized_tags.items():
                for tag in tags:
                    self.tag_counts[tag_type][tag] += 1
        
        self._compute_weights()
        
        if self.config.tag_weighting.use_cache:
            self._save_cache()

    def _compute_weights(self) -> None:
        """Compute tag weights based on frequency with proper scaling."""
        for tag_type in self.tag_counts:
            type_counts = self.tag_counts[tag_type]
            total_type_count = sum(type_counts.values())
            
            if total_type_count == 0:
                continue
                
            for tag, count in type_counts.items():
                # Calculate normalized frequency (0 to 1 range)
                frequency = count / self.total_samples
                
                # Apply inverse frequency with smoothing
                raw_weight = 1.0 / (frequency + self.smoothing_factor)
                
                # Scale the weight to our desired range
                # First normalize to 0-1 range
                max_possible_weight = 1.0 / self.smoothing_factor  # Theoretical maximum when frequency = 0
                normalized_weight = (raw_weight - 1.0) / (max_possible_weight - 1.0)
                
                # Then scale to our desired range
                scaled_weight = (
                    self.min_weight +
                    (normalized_weight * (self.max_weight - self.min_weight))
                )
                
                # Final clamp to ensure bounds
                weight = min(max(scaled_weight, self.min_weight), self.max_weight)
                
                # Save the computed weight for this tag
                self.tag_weights[tag_type][tag] = float(weight)

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

    def get_caption_weight_details(self, caption: str) -> Dict[str, any]:
        """Get detailed weights for all tags in a caption."""
        categorized_tags = self._extract_tags(caption)
        tag_details = {
            "total_weight": self.default_weight,
            "tags": defaultdict(list)
        }
        
        weights = []
        
        # Process each tag category
        for tag_type, tags in categorized_tags.items():
            tag_weights = []
            for tag in tags:
                weight = self.tag_weights[tag_type][tag]
                tag_weights.append(weight)
                tag_details["tags"][tag_type].append({
                    "tag": tag,
                    "weight": weight,
                    "frequency": self.tag_counts[tag_type][tag] / self.total_samples if self.total_samples > 0 else 0
                })
            
            if tag_weights:
                weights.append(np.mean(tag_weights))
        
        # Calculate total caption weight using geometric mean
        if weights:
            tag_details["total_weight"] = float(np.exp(np.mean(np.log(weights))))
        
        return dict(tag_details)

    def get_batch_weights(self, captions: List[str]) -> torch.Tensor:
        """Get weights for a batch of captions."""
        weights = [self.get_caption_weight(caption) for caption in captions]
        return torch.tensor(weights, dtype=torch.float32)

    def _save_cache(self) -> None:
        """Save tag statistics to cache."""
        cache_data = {
            "tag_counts": {k: dict(v) for k, v in self.tag_counts.items()},
            "tag_weights": {k: dict(v) for k, v in self.tag_weights.items()},
            "total_samples": self.total_samples
        }
        
        # Use cache manager to save if available
        if hasattr(self.config, 'cache_manager'):
            self.config.cache_manager.save_tag_weights(cache_data, self.cache_path)

    def _load_cache(self) -> None:
        """Load tag statistics from cache."""
        if not hasattr(self.config, 'cache_manager'):
            return
        
        cache_data = self.config.cache_manager.load_tag_weights(
            self.config.cache_manager.get_cache_key(self.cache_path)
        )
        if not cache_data:
            return
        
        self.tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_weights = defaultdict(lambda: defaultdict(lambda: self.default_weight))
        
        for tag_type, counts in cache_data["tag_counts"].items():
            self.tag_counts[tag_type].update(counts)
        
        for tag_type, weights in cache_data["tag_weights"].items():
            self.tag_weights[tag_type].update(weights)
        
        self.total_samples = cache_data["total_samples"]

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
                )[:3],
                "weight_range": (
                    min(self.tag_weights[tag_type].values()),
                    max(self.tag_weights[tag_type].values())
                )
            }
            stats[tag_type] = type_stats
            
        return stats

    def save_to_index(self, output_path: Path, image_tags: Dict[str, Dict[str, any]]) -> None:
        """Save tag weights and statistics to index using cache manager."""
        if not hasattr(self.config, 'cache_manager'):
            return
        
        index_data = self._prepare_index_data(image_tags)
        self.config.cache_manager.save_tag_index(index_data, output_path)

    def _prepare_index_data(self, image_tags: Dict[str, Dict[str, any]]) -> Dict[str, Any]:
        """Prepare index data structure without file operations."""
        index_data = {
            "metadata": {
                "total_samples": self.total_samples,
                "default_weight": self.default_weight,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "smoothing_factor": self.smoothing_factor,
                "tag_types": self.tag_types
            },
            "statistics": {
                "tag_counts": {k: dict(v) for k, v in self.tag_counts.items()},
                "tag_weights": {k: dict(v) for k, v in self.tag_weights.items()},
                "type_statistics": self.get_tag_statistics()
            },
            "images": {
                str(image_path): self._prepare_image_data(image_data)
                for image_path, image_data in image_tags.items()
            }
        }
        
        return self._clean_numeric_values(index_data)

    def _prepare_image_data(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare individual image data for index."""
        return {
            "caption": image_data["caption"],
            "total_weight": image_data["weight_details"]["total_weight"],
            "tags": {
                tag_type: [
                    {
                        "tag": tag_info["tag"],
                        "weight": float(tag_info["weight"]),
                        "frequency": float(tag_info["frequency"])
                    }
                    for tag_info in tags_list
                ]
                for tag_type, tags_list in image_data["weight_details"]["tags"].items()
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

    def process_dataset_tags(self, image_captions: Dict[str, str]) -> Dict[str, Dict[str, any]]:
        """Process all image captions and return detailed tag information."""
        image_tags = {}
        
        for image_path, caption in image_captions.items():
            # Get detailed tag analysis
            tag_details = self.get_caption_weight_details(caption)
            
            # Store comprehensive information
            image_tags[image_path] = {
                "caption": caption,
                "weight_details": {
                    "total_weight": tag_details["total_weight"],
                    "tags": {
                        tag_type: [
                            {
                                "tag": tag_info["tag"],
                                "weight": tag_info["weight"],
                                "frequency": tag_info["frequency"]
                            }
                            for tag_info in tags_list
                        ]
                        for tag_type, tags_list in tag_details["tags"].items()
                    }
                }
            }
        
        return image_tags

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
    index_output_path: Path,
    cache_path: Optional[Path] = None
) -> TagWeighter:
    """Create a tag weighter and save comprehensive index."""
    weighter = TagWeighter(config, cache_path)
    
    # Process all captions and compute weights if not cached
    if not (config.tag_weighting.use_cache and weighter.cache_path.exists()):
        logger.info("Computing tag statistics...")
        weighter.update_statistics(list(image_captions.values()))
        weighter._save_cache()
    
    # Process and save individual image tags
    logger.info("Processing image tags...")
    image_tags = weighter.process_dataset_tags(image_captions)
    
    # Save index data using cache manager
    if hasattr(config, 'cache_manager'):
        index_data = weighter._prepare_index_data(image_tags)
        config.cache_manager.save_tag_index(index_data, index_output_path)
    
    return weighter

def preprocess_dataset_tags(
    config: "Config",
    image_paths: List[str],
    captions: List[str],
    cache_dir: Optional[Path] = None
) -> Optional[TagWeighter]:
    """Preprocess all dataset tags before training.
    
    Returns:
        TagWeighter instance if tag weighting is enabled, None otherwise.
    """
    if not config.tag_weighting.enable_tag_weighting:
        return None
        
    logger.info("Starting tag preprocessing...")
    
    # Setup paths
    if cache_dir is None:
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
    cache_dir = Path(cache_dir)
    
    # Create and initialize tag weighter
    image_captions = dict(zip(image_paths, captions))
    index_path = cache_dir / "tag_weights_index.json"
    
    logger.info("Processing tags and creating index...")
    weighter = create_tag_weighter_with_index(
        config=config,
        image_captions=image_captions,
        index_output_path=index_path
    )
    
    logger.info("Tag preprocessing complete")
    return weighter
