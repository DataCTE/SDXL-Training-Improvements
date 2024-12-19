"""Tag weighting system for SDXL training."""
import json
from collections import defaultdict
from src.core.logging.logging import setup_logging
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.data.config import Config

logger = setup_logging(__name__, level="INFO")

class TagWeighter:
    def __init__(
        self,
        config: "Config",  # type: ignore
        cache_path: Optional[Path] = None
    ):
        """Initialize tag weighting system."""
        self.config = config
        from src.utils.paths import convert_windows_path
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir, make_absolute=True)
        self.cache_path = cache_path or Path(convert_windows_path(Path(cache_dir) / "tag_weights.json", make_absolute=True))
        
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
        import re
        tags = [t.strip() for t in re.split(r'[,\(\)]', caption) if t.strip()]
        
        categorized_tags = defaultdict(list)
        for tag in tags:
            tag_type = self._determine_tag_type(tag)
            categorized_tags[tag_type].append(tag)
            
        return dict(categorized_tags)

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
        """Compute tag weights based on frequency."""
        for tag_type in self.tag_counts:
            type_counts = self.tag_counts[tag_type]
            total_type_count = sum(type_counts.values())
            
            if total_type_count == 0:
                continue
                
            for tag, count in type_counts.items():
                frequency = count / self.total_samples
                weight = 1.0 / (frequency + self.smoothing_factor)
                weight = np.clip(weight, self.min_weight, self.max_weight)
                self.tag_weights[tag_type][tag] = float(weight)

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
        
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f)

    def _load_cache(self) -> None:
        """Load tag statistics from cache."""
        with open(self.cache_path, 'r') as f:
            cache_data = json.load(f)
            
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
                )[:10],
                "weight_range": (
                    min(self.tag_weights[tag_type].values()),
                    max(self.tag_weights[tag_type].values())
                )
            }
            stats[tag_type] = type_stats
            
        return stats

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
