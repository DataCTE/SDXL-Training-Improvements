"""Tag weighting system for SDXL training with JSON index support."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any
import numpy as np
import torch
import time
import spacy
from tqdm import tqdm
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from src.data.config import Config

from src.core.logging import UnifiedLogger, LogConfig, ProgressPredictor, get_logger, setup_logging
from src.data.utils.paths import convert_windows_path
from src.models.sdxl import StableDiffusionXL
from src.models.encoders import CLIPEncoder
from src.data.preprocessing.exceptions import TagProcessingError
from src.data.preprocessing.cache_manager import CacheManager

logger = setup_logging(
    LogConfig(
        name=__name__,
        enable_progress=True,
        enable_metrics=True,
        enable_memory=True
    )
)

def default_int():
    return 0

class DefaultDict:
    def __init__(self, default_factory):
        self.default_factory = default_factory
        self.data = {}
        
    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.default_factory()
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __contains__(self, key):
        return key in self.data
        
    def items(self):
        return self.data.items()
        
    def values(self):
        return self.data.values()
        
    def keys(self):
        return self.data.keys()
        
    def update(self, other):
        self.data.update(other)
        
    def get(self, key, default=None):
        return self.data.get(key, default)
        
    def __iter__(self):
        return iter(self.data)
        
    def __len__(self):
        return len(self.data)

class TagWeighter:
    def __init__(
        self,
        config: "Config",  # type: ignore
        model: Optional["StableDiffusionXL"] = None,  # Parameter kept for compatibility
        cache_manager: Optional["CacheManager"] = None
    ):
        """Initialize tag weighting system."""
        self.config = config
        
        # Core settings
        self.default_weight = config.tag_weighting.default_weight
        self.min_weight = config.tag_weighting.min_weight
        self.max_weight = config.tag_weighting.max_weight
        self.smoothing_factor = config.tag_weighting.smoothing_factor
        
        # Initialize tag categories
        self.tag_types = {
            "subject": [],  # Will be populated dynamically
            "style": [],
            "quality": [],
            "technical": [],
            "meta": []
        }
        
        # Initialize counters with proper defaults
        self.tag_counts = {}
        self.tag_weights = {}
        for tag_type in self.tag_types:
            self.tag_counts[tag_type] = DefaultDict(default_int)
            self.tag_weights[tag_type] = DefaultDict(self.get_default_weight)
            
        self.total_samples = 0
        
        # Cache reference - create new if not provided
        if cache_manager is None:
            cache_manager = CacheManager(
                cache_dir=config.global_config.cache.cache_dir,
                config=config,
                max_cache_size=config.global_config.cache.max_cache_size
            )
        self.cache_manager = cache_manager
        
        # NLP will be initialized when needed
        self._nlp = None

    def get_default_weight(self):
        """Return default weight value."""
        return self.default_weight

    def verify_cache_validity(self) -> bool:
        """Verify that loaded cache data is valid and complete."""
        try:
            # Check basic structure
            if not self.tag_counts or not self.tag_weights:
                logger.warning("Missing tag counts or weights")
                return False
                
            # Verify all tag types are present
            for tag_type in self.tag_types:
                if tag_type not in self.tag_counts:
                    logger.warning(f"Missing tag counts for type: {tag_type}")
                    return False
                if tag_type not in self.tag_weights:
                    logger.warning(f"Missing tag weights for type: {tag_type}")
                    return False
                    
            # Verify weight ranges
            for tag_type, weights in self.tag_weights.items():
                if not weights:  # Skip empty categories
                    continue
                min_weight = min(weights.values())
                max_weight = max(weights.values())
                if not (self.min_weight <= min_weight <= max_weight <= self.max_weight):
                    logger.warning(
                        f"Invalid weight range for {tag_type}: "
                        f"{min_weight} to {max_weight}, expected "
                        f"{self.min_weight} to {self.max_weight}"
                    )
                    return False
                    
            # Verify total samples
            if self.total_samples <= 0:
                logger.warning(f"Invalid total samples count: {self.total_samples}")
                return False
                
            logger.info(
                f"Cache validated: {len(self.tag_counts)} types, "
                f"{self.total_samples} samples"
            )
            return True
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False

    def _get_tag_category(self, tag: str) -> str:
        """Determine tag category using semantic understanding with spaCy NLP.
        
        Categories:
        - subject: The main focus/subject matter (people, objects, scenes, etc)
        - style: Artistic style, medium, or aesthetic qualities
        - quality: Image quality attributes and specifications
        - technical: Technical aspects of composition and photography
        - meta: Metadata and other administrative tags
        """
        try:
            tag = tag.lower().strip()
            
            # Check for explicit category prefix
            if ":" in tag:
                try:
                    category = tag.split(":")[0]
                    if category in self.tag_types:
                        return category
                except Exception as e:
                    logger.warning(f"Error parsing category prefix: {e}")
                    
            # Remove common tag prefixes/suffixes for cleaner analysis
            tag = tag.replace("_", " ").strip()
            
            try:
                # Parse with spaCy
                doc = self._nlp(tag)
            except Exception as e:
                logger.error(f"SpaCy parsing failed for tag '{tag}': {e}")
                return "meta"  # Default to meta on parsing failure

            # Extract linguistic features
            has_subject = any(token.dep_ in ['nsubj', 'dobj'] for token in doc)
            has_location = any(token.dep_ == 'pobj' for token in doc)
            has_action = any(token.pos_ == 'VERB' for token in doc)
            has_quality = any(token.pos_ == 'ADJ' for token in doc)
            has_technical = any(token.like_num or token.text.endswith(('k', 'p', 'fps')) for token in doc)

            # Style-specific patterns
            style_suffixes = ('ism', 'esque', 'like', 'tone', 'color', 'shade')
            has_style = any(token.text.endswith(style_suffixes) for token in doc)
            
            # Technical photography terms
            tech_terms = {'close', 'wide', 'depth', 'field', 'ratio', 'light', 'shot', 'view', 'angle'}
            has_tech_term = any(token.text in tech_terms for token in doc)
            
            # Determine category based on linguistic features
            if has_subject or (has_action and not has_technical):
                return "subject"
            if has_style or any(ent.label_ == 'WORK_OF_ART' for ent in doc.ents):
                return "style"
            if has_technical or has_tech_term:
                return "technical"
            if has_quality and not (has_subject or has_style):
                return "quality"
            if has_location and not has_subject:
                return "subject"  # Locations are treated as subjects
            
            # Default to meta for organizational/administrative tags
            return "meta"
            
        except Exception as e:
            logger.error(f"Error in tag categorization: {e}")
            return "meta"

    def _init_nlp(self):
        """Initialize NLP model lazily."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'lemmatizer'])
            except OSError:
                raise ImportError(
                    "SpaCy model 'en_core_web_sm' not found. "
                    "Please install it with: python -m spacy download en_core_web_sm"
                )
        return self._nlp

    @staticmethod
    def _init_worker():
        """Initialize worker process with spaCy model."""
        global nlp
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise

    def _process_caption_batch(self, captions: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of captions."""
        try:
            results = []
            # Initialize NLP if needed
            self._init_nlp()
            
            for caption in captions:
                tags = self._extract_tags(caption)
                if tags:
                    results.append(tags)
            return results
        except Exception as e:
            logger.error(f"Failed to process caption batch: {e}")
            return []

    def process_dataset_tags(self, captions: List[str]) -> Dict[str, Dict[str, Any]]:
        """Process dataset tags sequentially with batching for memory efficiency."""
        logger.info("Starting tag processing...")
        
        # Process in batches for memory efficiency
        batch_size = 1000
        batches = [captions[i:i + batch_size] for i in range(0, len(captions), batch_size)]
        
        # Initialize results dictionary
        results = {}
        
        # Process each batch
        for i, batch in enumerate(batches):
            try:
                # Process each caption in the batch
                for caption_idx, caption in enumerate(batch):
                    global_idx = i * batch_size + caption_idx
                    tags = self._extract_tags(caption)
                    if tags:
                        # Store results with weights
                        results[str(global_idx)] = {
                            "tags": {
                                tag_type: [
                                    {
                                        "tag": tag,
                                        "weight": self.tag_weights[tag_type][tag]
                                    }
                                    for tag in tags[tag_type]
                                ]
                                for tag_type in tags
                            }
                        }
                
                # Log progress
                progress = ((i + 1) / len(batches)) * 100
                logger.info(f"Processing batch {i+1}/{len(batches)} ({progress:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i}: {e}")
                continue
        
        return results

    def _process_weights_batch(self, captions: List[str], batch_tags: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Process weights for a batch of captions."""
        batch_results = {}
        
        # Create a local copy of weights for this batch
        local_weights = {}
        for tag_type in self.tag_types:
            local_weights[tag_type] = dict(self.tag_weights[tag_type].items())
            
        for caption in captions:
            tags = batch_tags[caption]
            weighted_tags = {
                tag_type: [
                    {
                        "tag": tag, 
                        "weight": local_weights[tag_type].get(tag, self.default_weight)
                    }
                    for tag in set(tags[tag_type])
                ]
                for tag_type in tags
            }
            
            # Calculate caption weight using local weights
            weights = []
            for tag_type, tag_list in tags.items():
                if not tag_list:
                    continue
                
                unique_tags = set(tag_list)
                type_weights = [
                    local_weights[tag_type].get(tag, self.default_weight)
                    for tag in unique_tags
                ]
                
                if type_weights:
                    weights.append(float(np.mean(type_weights)))
            
            caption_weight = (
                float(np.exp(np.mean(np.log(weights))))
                if weights else self.default_weight
            )
            
            batch_results[caption] = {
                "tags": weighted_tags,
                "weight": caption_weight
            }
            
        return batch_results

    def update_statistics(self, captions: List[str]) -> None:
        """Update tag statistics from captions using parallel processing."""
        try:
            logger.info("Starting tag statistics update...")
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            batches = [captions[i:i + batch_size] for i in range(0, len(captions), batch_size)]
            
            # Process each batch
            for i, batch in enumerate(batches):
                try:
                    # Process tags for this batch
                    for caption in batch:
                        tags = self._extract_tags(caption)
                        # Update counts for each tag type
                        for tag_type, tag_list in tags.items():
                            for tag in tag_list:
                                self.tag_counts[tag_type][tag] += 1
                    
                    # Log progress
                    progress = ((i + 1) / len(batches)) * 100
                    logger.info(f"Processing batch {i+1}/{len(batches)} ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {i}: {e}")
                    continue
            
            # Update total samples
            self.total_samples += len(captions)
            
            # Compute weights after all processing is done
            self._compute_weights()
            
            # Save to cache if enabled
            if self.config.tag_weighting.use_cache:
                self.save_to_cache()
                
        except Exception as e:
            logger.error(f"Failed to update tag statistics: {e}")
            raise TagProcessingError("Failed to update tag statistics", context={"error": str(e)})

    def _compute_weights(self) -> None:
        """Compute tag weights using vectorized operations."""
        # Pre-compute constants
        min_max_diff = self.max_weight - self.min_weight
        max_possible_weight = 1.0 / self.smoothing_factor
        weight_range = max_possible_weight - 1.0
        
        for tag_type in self.tag_counts:
            type_counts = dict(self.tag_counts[tag_type].items())
            if not type_counts:
                continue
            
            # Convert to numpy arrays for vectorized operations
            tags = list(type_counts.keys())
            counts = np.array(list(type_counts.values()), dtype=np.float32)
            
            # Calculate frequencies
            total_count = np.sum(counts)
            if total_count == 0:
                continue
                
            frequencies = counts / total_count
            
            # Compute weights in a single vectorized operation
            weights = np.minimum(
                np.maximum(
                    self.min_weight + (
                        ((1.0 / (frequencies + self.smoothing_factor)) - 1.0)
                        / weight_range * min_max_diff
                    ),
                    self.min_weight
                ),
                self.max_weight
            )
            
            # Update weights dictionary with computed values
            computed_weights = dict(zip(tags, weights.tolist()))
            for tag, weight in computed_weights.items():
                self.tag_weights[tag_type][tag] = weight

    def get_caption_weight(self, caption: str) -> float:
        """Get combined weight for a caption using vectorized operations."""
        try:
            categorized_tags = self._extract_tags(caption)
            weights = []
            
            # Pre-allocate arrays for better performance
            for tag_type, tags in categorized_tags.items():
                if not tags:
                    continue
                    
                # Get unique tags and their weights
                unique_tags = set(tags)
                type_weights = np.array([
                    self.tag_weights[tag_type][tag] 
                    for tag in unique_tags
                ], dtype=np.float32)
                
                if len(type_weights) > 0:
                    # Use numpy mean for better performance
                    weights.append(np.mean(type_weights))
            
            if not weights:
                return self.default_weight
            
            # Compute geometric mean using numpy
            weights_array = np.array(weights, dtype=np.float32)
            return float(np.exp(np.mean(np.log(weights_array))))
                
        except Exception as e:
            logger.error(f"Error calculating caption weight: {e}")
            return self.default_weight

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
                logger.warning("No cache manager for loading cache")
                return False
            
            logger.info("Loading tag index from cache manager...")
            index_data = self.cache_manager.load_tag_index()
            if not index_data:
                logger.warning("No index data found")
                return False
                
            if not self._validate_tag_data(index_data):
                logger.warning("Tag data validation failed")
                return False
            
            # Load statistics from index
            logger.info("Loading statistics from index...")
            if "statistics" in index_data:
                stats = index_data["statistics"]
                if "tag_counts" in stats:
                    for tag_type, counts in stats["tag_counts"].items():
                        self.tag_counts[tag_type].update(counts)
                    logger.info(f"Loaded tag counts for {len(stats['tag_counts'])} types")
                
                if "tag_weights" in stats:
                    for tag_type, weights in stats["tag_weights"].items():
                        self.tag_weights[tag_type].update(weights)
                    logger.info(f"Loaded tag weights for {len(stats['tag_weights'])} types")
            
            # Load metadata
            if "metadata" in index_data:
                self.total_samples = index_data["metadata"].get("total_samples", 0)
                logger.info(f"Loaded metadata with {self.total_samples} total samples")
            
            # Verify loaded data
            if not self.verify_cache_validity():
                logger.warning("Cache validation failed")
                return False
                
            logger.info("Successfully loaded and validated tag cache")
            return True
            
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
        # Don't pickle the NLP model
        state['_nlp'] = None
        return state

    def __setstate__(self, state):
        """Customize unpickling behavior."""
        self.__dict__.update(state)
        # NLP will be reinitialized when needed

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
                logger.warning("No cache manager available")
                return False
            
            if self.config.tag_weighting.use_cache:
                # Try loading from cache first
                logger.info("Attempting to load tag cache...")
                if self._load_cache():
                    # Verify cache contents
                    logger.info("Cache loaded, verifying tag index...")
                    tag_index = self.cache_manager.load_tag_index()
                    if not tag_index:
                        logger.warning("Tag index is empty")
                        return False
                    if "images" not in tag_index:
                        logger.warning("Tag index missing 'images' section")
                        return False
                    if "statistics" not in tag_index:
                        logger.warning("Tag index missing 'statistics' section")
                        return False
                        
                    logger.info(f"Tag index verified with {len(tag_index['images'])} images")
                    return True
                else:
                    logger.warning("Failed to load tag cache data")
                    return False
                        
            logger.info("Tag cache disabled in config")
            return False
            
        except Exception as e:
            logger.error(f"Tag system initialization failed: {e}")
            return False

    def _validate_tag_data(self, tag_data: Dict[str, Any]) -> bool:
        """Validate loaded tag data structure and content."""
        try:
            # Check version
            if "version" not in tag_data:
                return False
            
            # Check required sections
            required_sections = ["metadata", "statistics"]
            if not all(section in tag_data for section in required_sections):
                return False
            
            # Validate statistics
            if not isinstance(tag_data["statistics"], dict):
                return False
            
            # Validate tag counts and weights exist
            stats = tag_data["statistics"]
            if "tag_counts" not in stats or "tag_weights" not in stats:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Tag data validation failed: {e}")
            return False

    def _extract_tags(self, caption: str) -> Dict[str, List[str]]:
        """Extract and categorize tags from caption efficiently."""
        # Initialize NLP if needed
        self._init_nlp()
        
        categorized = {
            "subject": [], "style": [], "quality": [], 
            "technical": [], "meta": []
        }
        
        # Quick split and clean
        tags = [t.strip() for t in caption.split(',') if t and len(t.strip()) <= 100]
        
        for tag in tags:
            # Handle explicit category prefix
            if ":" in tag:
                try:
                    category, clean_tag = tag.split(":", 1)
                    if category in categorized:
                        categorized[category].append(clean_tag.strip())
                        continue
                except Exception:
                    pass
            
            # Clean tag for analysis
            clean_tag = tag.replace("_", " ").strip()
            
            # Quick pattern matching before using spaCy
            if any(suffix in clean_tag for suffix in ('ism', 'esque', 'like', 'tone', 'color', 'shade')):
                categorized["style"].append(tag)
            elif any(term in clean_tag for term in ('close', 'wide', 'depth', 'field', 'ratio', 'light', 'shot', 'view', 'angle')):
                categorized["technical"].append(tag)
            elif any(c.isdigit() for c in clean_tag) or any(clean_tag.endswith(x) for x in ('k', 'p', 'fps')):
                categorized["technical"].append(tag)
            else:
                try:
                    doc = self._nlp(clean_tag)
                    if any(token.dep_ in ['nsubj', 'dobj'] for token in doc):
                        categorized["subject"].append(tag)
                    elif any(token.pos_ == 'ADJ' for token in doc):
                        categorized["quality"].append(tag)
                    else:
                        categorized["meta"].append(tag)
                except Exception:
                    categorized["meta"].append(tag)
        
        return categorized

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
    captions: List[str],
    model: Optional["StableDiffusionXL"] = None,
    cache_manager: Optional["CacheManager"] = None
) -> TagWeighter:
    """Create and initialize tag weighter with index."""
    # Create cache manager if not provided
    if cache_manager is None:
        cache_manager = CacheManager(
            cache_dir=config.global_config.cache.cache_dir,
            config=config,
            max_cache_size=config.global_config.cache.max_cache_size
        )
    
    # Initialize tag weighter with cache manager
    weighter = TagWeighter(config, model=model, cache_manager=cache_manager)
    
    logger.info("Processing captions and updating tag statistics...")
    weighter.update_statistics(captions)
    
    logger.info("Creating detailed tag index...")
    image_tags = weighter.process_dataset_tags(captions)
    
    # Save to cache manager's tag directory
    logger.info("Saving tag index to cache")
    weighter.cache_manager.save_tag_index({
        "version": "1.0",
        "metadata": {
            "total_samples": weighter.total_samples,
            "created_at": time.time()
        },
        "statistics": {
            "tag_counts": {
                tag_type: dict(counts.items())
                for tag_type, counts in weighter.tag_counts.items()
            },
            "tag_weights": {
                tag_type: dict(weights.items())
                for tag_type, weights in weighter.tag_weights.items()
            }
        },
        "images": image_tags  # Now directly using the dictionary
    })
    
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
    logger.info("Processing tags and creating index...")
    weighter = create_tag_weighter_with_index(
        config=config,
        captions=captions
    )
    
    logger.info("Tag preprocessing complete")
    return weighter
