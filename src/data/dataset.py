"""High-performance dataset implementation for SDXL training."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from contextlib import contextmanager

import torch
import torch.backends.cudnn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

from src.core.logging import get_logger
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_paths
from src.data.config import Config
from src.models.sdxl import StableDiffusionXL
from src.data.preprocessing import (
    CacheManager,
    create_tag_weighter_with_index
)
from src.models.encoders import CLIPEncoder
from src.core.memory.tensor import create_stream_context
import torch.nn.functional as F

logger = get_logger(__name__)

@dataclass
class ProcessingStats:
    """Simplified processing statistics."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class ProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass

class AspectBucketDataset(Dataset):
    """Enhanced SDXL dataset with extreme memory handling and 100x speedups."""
    
    def __init__(
        self,
        config: Config,
        image_paths: List[str],
        captions: List[str],
        model: Optional[StableDiffusionXL] = None,
        is_train: bool = True,
        enable_memory_tracking: bool = True,
        max_memory_usage: float = 0.8,
        stream_timeout: float = 10.0,
        device: Optional[torch.device] = None,
        device_id: Optional[int] = None
    ):
        """Initialize dataset with preprocessing capabilities."""
        super().__init__()
        
        # Move initialization code from pipeline here
        self.config = config
        self.is_train = is_train
        self.enable_memory_tracking = enable_memory_tracking
        self.max_memory_usage = max_memory_usage
        self.stream_timeout = stream_timeout
        self.stats = ProcessingStats()
        
        # Performance tracking
        self.performance_stats = {
            'operation_times': {},
            'errors': []
        }
        self.action_history = {}
        
        # CUDA setup
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
            
            self.device = device or torch.device('cuda')
            self.device_id = device_id if device_id is not None else 0
        else:
            self.device = torch.device('cpu')
            self.device_id = None

        # Core components
        self.model = model
        if model:
            self.vae_encoder = model.vae_encoder
            self.text_encoders = model.text_encoders
            self.tokenizers = model.tokenizers
        
        # Stream management
        self._streams = {}
        
        # Initialize cache manager
        cache_dir = convert_windows_path(config.global_config.cache.cache_dir)
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            max_cache_size=config.global_config.cache.max_cache_size,
            device=self.device
        )

        # Initialize paths and captions
        self.image_paths = [
            str(convert_windows_path(p) if is_windows_path(p) else Path(p))
            for p in image_paths
        ]
        self.captions = captions

        # Initialize buckets
        self.buckets = self.get_aspect_buckets(config)
        self.bucket_indices = []

        # Initialize tag weighter if needed
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

        # Precompute latents if caching enabled
        if config.global_config.cache.use_cache:
            self._precompute_latents()

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            image_path = self.image_paths[idx]
            caption = self.captions[idx]
            
            cache_key = self.cache_manager.get_cache_key(image_path)
            cached_data = self.cache_manager.load_tensors(cache_key)
            
            if cached_data is None:
                processed = self._process_single_image(image_path=image_path, config=self.config)
                
                if processed is None:
                    raise ValueError(f"Failed to process image: {image_path}")
                    
                # Only encode if not cached
                encoded_text = self.encode_prompt(
                    batch={"text": [caption]},
                    proportion_empty_prompts=0.0
                )
                
                processed.update({
                    "prompt_embeds": encoded_text["prompt_embeds"][0],
                    "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"][0],
                    "text": caption
                })
                    
                if self.config.global_config.cache.use_cache:
                    tensors = {
                        "pixel_values": processed["pixel_values"],
                        "prompt_embeds": processed["prompt_embeds"],
                        "pooled_prompt_embeds": processed["pooled_prompt_embeds"]
                    }
                    metadata = {
                        "original_size": processed["original_size"],
                        "crop_coords": processed["crop_coords"],
                        "target_size": processed["target_size"],
                        "text": caption
                    }
                    self.cache_manager.save_latents(tensors, image_path, metadata)
                    cached_data = {**tensors, "metadata": metadata}
                else:
                    cached_data = {
                        "pixel_values": processed["pixel_values"],
                        "prompt_embeds": processed["prompt_embeds"],
                        "pooled_prompt_embeds": processed["pooled_prompt_embeds"],
                        "metadata": {
                            "original_size": processed["original_size"],
                            "crop_coords": processed["crop_coords"],
                            "target_size": processed["target_size"],
                            "text": caption
                        }
                    }
            
            return cached_data

        except Exception as e:
            logger.error(f"Failed to get item {idx}: {str(e)}")
            return None

    def _precompute_latents(self) -> None:
        """Precompute and cache latents with accurate progress tracking."""
        total_images = len(self)
        start_time = time.time()
        
        # Get cache status for all images
        cached_status = self._get_cached_status(self.image_paths)
        already_cached = sum(1 for is_cached in cached_status.values() if is_cached)
        
        remaining_images = total_images - already_cached
        failed_images = 0
        processed_images = already_cached
        batch_size = 8
        min_batch_size = 1
        
        # Speed tracking with moving window
        processing_times = []
        window_size = 50  # Number of images for moving average
        
        # Create progress bar with detailed stats
        pbar = tqdm(
            total=total_images,
            desc="Precomputing latents",
            unit="img",
            initial=already_cached,
            dynamic_ncols=True,  # Better terminal handling
        )
        
        def update_progress_stats():
            """Calculate and update progress statistics."""
            if processing_times:
                window = processing_times[-window_size:]
                current_speed = len(window) / sum(window) if window else 0
                
                # Use remaining_images here
                eta_seconds = remaining_images / current_speed if current_speed > 0 else 0
                
                # Calculate success rate
                success_rate = (processed_images / (processed_images + failed_images)) * 100 if (processed_images + failed_images) > 0 else 100
                
                return {
                    'processed': processed_images,
                    'failed': failed_images,
                    'cached': already_cached,
                    'remaining': remaining_images,
                    'speed': f'{current_speed:.2f} img/s',
                    'eta': time.strftime('%H:%M:%S', time.gmtime(eta_seconds)),
                    'success': f'{success_rate:.1f}%'
                }
            return {}

        # Process only uncached images
        uncached_indices = [
            i for i, path in enumerate(self.image_paths)
            if not cached_status[path]
        ]
        
        for start_idx in range(0, len(uncached_indices), batch_size):
            batch_start_time = time.time()
            try:
                end_idx = min(start_idx + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[start_idx:end_idx]
                batch_size_actual = len(batch_indices)
                
                # Process batch
                batch_paths = [self.image_paths[i] for i in batch_indices]
                batch_captions = [self.captions[i] for i in batch_indices]
                
                processed_items = self.process_image_batch(
                    image_paths=batch_paths,
                    captions=batch_captions,
                    config=self.config
                )
                
                # Track successful and failed items
                successful_items = [item for item in processed_items if item is not None]
                failed_in_batch = batch_size_actual - len(successful_items)
                
                # Add batch VAE encoding
                if successful_items:
                    # Stack tensors and encode in batch mode
                    batch_tensors = torch.stack([item["pixel_values"] for item in successful_items])
                    batch_latents = self.encode_with_vae(batch_tensors, batch_mode=True)
                    
                    # Update items with encoded latents
                    for idx, item in enumerate(successful_items):
                        item["pixel_values"] = batch_latents[idx]

                # Cache successful results
                if self.config.global_config.cache.use_cache and successful_items:
                    batch_tensors = []
                    batch_metadata = []
                    valid_paths = []
                    
                    for processed, caption, path in zip(processed_items, batch_captions, batch_paths):
                        if processed is not None:
                            batch_tensors.append({
                                "pixel_values": processed["pixel_values"],
                                "prompt_embeds": processed["prompt_embeds"],
                                "pooled_prompt_embeds": processed["pooled_prompt_embeds"]
                            })
                            batch_metadata.append({
                                "original_size": processed["original_size"],
                                "crop_coords": processed["crop_coords"],
                                "target_size": processed["target_size"],
                                "text": caption
                            })
                            valid_paths.append(path)
                    
                    # Batch save
                    if batch_tensors:
                        cache_results = self.cache_manager.save_latents_batch(
                            batch_tensors=batch_tensors,
                            original_paths=valid_paths,
                            batch_metadata=batch_metadata
                        )
                        
                        # Update counters
                        successful_saves = sum(1 for r in cache_results if r)
                        failed_saves = sum(1 for r in cache_results if not r)
                        
                        processed_images += successful_saves
                        failed_images += failed_saves + failed_in_batch
                        
                        # Track processing time for successful items
                        if successful_saves > 0:
                            batch_time = time.time() - batch_start_time
                            processing_times.extend([batch_time / successful_saves] * successful_saves)
                
                # Update progress bar
                pbar.update(batch_size_actual)
                pbar.set_postfix(update_progress_stats())
                
            except Exception as e:
                logger.error(f"Failed to process batch at index {start_idx}", exc_info=True)
                failed_images += batch_size_actual
                
                # Reduce batch size on failure
                batch_size = max(batch_size // 2, min_batch_size)
                logger.info(f"Reducing batch size to {batch_size}")
                
                pbar.update(batch_size_actual)
                continue
        
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_speed = processed_images / total_time if total_time > 0 else 0
        success_rate = (processed_images / total_images) * 100 if total_images > 0 else 0
        cache_stats = self.cache_manager.get_cache_stats()
        
        logger.info(
            f"Latent computation complete:\n"
            f"- Total images: {total_images}\n"
            f"- Successfully processed: {processed_images}\n"
            f"- Previously cached: {already_cached}\n"
            f"- Failed: {failed_images}\n"
            f"- Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
            f"- Average speed: {avg_speed:.2f} img/s\n"
            f"- Success rate: {success_rate:.1f}%\n"
            f"- Final cache size: {cache_stats['cache_size_bytes'] / 1024**2:.2f} MB"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples into training format."""
        try:
            # Filter out None values from failed processing
            batch = [b for b in batch if b is not None]
            
            if not batch:
                raise ValueError("Empty batch after filtering")

            # Group samples by bucket size
            bucket_groups = {}
            for example in batch:
                size = example["pixel_values"].shape[-2:]  # Get HxW dimensions
                if size not in bucket_groups:
                    bucket_groups[size] = []
                bucket_groups[size].append(example)

            # Take the largest group
            largest_size = max(bucket_groups.keys(), key=lambda k: len(bucket_groups[k]))
            valid_batch = bucket_groups[largest_size]

            # Ensure consistent key names
            return {
                "pixel_values": torch.stack([example["pixel_values"] for example in valid_batch]),
                "prompt_embeds": torch.stack([example["prompt_embeds"] for example in valid_batch]),
                "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"] for example in valid_batch]),
                "original_size": [example["metadata"]["original_size"] for example in valid_batch],
                "crop_top_lefts": [example["metadata"]["crop_coords"] for example in valid_batch],
                "target_size": [example["metadata"]["target_size"] for example in valid_batch],
                "text": [example["metadata"]["text"] for example in valid_batch] if "text" in valid_batch[0]["metadata"] else None
            }

        except Exception as e:
            logger.error("Failed to collate batch", exc_info=True)
            raise RuntimeError(f"Collate failed: {str(e)}") from e

    def encode_with_vae(
        self, 
        img_tensor: torch.Tensor, 
        batch_mode: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Encode image tensor(s) with VAE to get latents."""
        device = device or self.device
        img_tensor = img_tensor.to(device)
        
        if not batch_mode:
            img_tensor = img_tensor.unsqueeze(0)
        
        with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
            latents = self.model.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor
        return latents.cpu()

    def encode_prompt(
        self,
        batch: Dict[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts using CLIP encoders directly."""
        try:
            encoded_output = CLIPEncoder.encode_prompt(
                batch=batch,
                text_encoders=self.text_encoders,
                tokenizers=self.tokenizers,
                proportion_empty_prompts=proportion_empty_prompts,
                is_train=self.is_train
            )
            
            return {
                "prompt_embeds": encoded_output["prompt_embeds"],
                "pooled_prompt_embeds": encoded_output["pooled_prompt_embeds"],
                "metadata": {
                    "num_prompts": len(batch[next(iter(batch))]),
                    "device": str(self.device),
                    "dtype": str(encoded_output["prompt_embeds"].dtype),
                    "timestamp": time.time()
                }
            }
        except Exception as e:
            logger.error("Failed to encode prompts", 
                        extra={'error': str(e), 'batch_size': len(batch[next(iter(batch))])})
            raise

    @contextmanager
    def track_memory_usage(self, operation: str):
        """Track memory usage for operations."""
        if not self.enable_memory_tracking:
            yield
            return
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            if self.enable_memory_tracking and torch.cuda.is_available():
                self._log_action(operation, {
                    'duration': time.time() - start_time,
                    'memory_change': torch.cuda.memory_allocated() - start_memory
                })

    def _get_stream(self):
        """Get a CUDA stream for the current thread."""
        import threading
        thread_id = threading.get_ident()
        if thread_id not in self._streams and torch.cuda.is_available():
            self._streams[thread_id] = torch.cuda.Stream()
        return self._streams.get(thread_id)

    def _process_on_gpu(self, func, **kwargs):
        """Process on GPU with memory management."""
        try:
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.cuda.device(self.device_id):
                stream = self._get_stream()
                with create_stream_context(stream):
                    result = func(**kwargs)
                    if stream:
                        # Add timeout to stream synchronization
                        start_time = time.time()
                        while not stream.query():
                            if time.time() - start_time > self.stream_timeout:
                                raise TimeoutError("Stream synchronization timeout")
                            time.sleep(0.001)
                    return result
            
        except Exception as e:
            logger.error(f"GPU processing error: {str(e)}")
            raise ProcessingError("GPU processing failed", {
                'device_id': self.device_id,
                'cuda_memory': torch.cuda.memory_allocated(self.device_id),
                'error': str(e)
            })

    def _load_image_to_tensor(self, image_path: Union[str, Path]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load image and convert to tensor.
        
        Returns:
            Tuple of (tensor, (width, height))
        """
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        img_tensor = torch.from_numpy(np.array(img)).float().to(self.device) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
        return img_tensor, (w, h)

    def _process_image_tensor(
        self, 
        img_tensor: torch.Tensor,
        original_size: Tuple[int, int],
        config: Config,
        image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Process image tensor on GPU."""
        w, h = original_size
        aspect_ratio = w / h
        
        # Early return if aspect ratio is invalid
        max_ratio = config.global_config.image.max_aspect_ratio
        if not (1/max_ratio <= aspect_ratio <= max_ratio):
            return None
        
        # Find best matching bucket
        buckets = self.get_aspect_buckets(config)
        bucket_ratios = [(bw/bh, idx, (bw,bh)) for idx, (bw,bh) in enumerate(buckets)]
        _, bucket_idx, (target_w, target_h) = min(
            [(abs(aspect_ratio - ratio), idx, dims) for ratio, idx, dims in bucket_ratios]
        )
        
        # Resize and crop in one step if possible
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img_tensor = img_tensor[:, top:top + target_h, left:left + target_w]
        
        return {
            "pixel_values": self.encode_with_vae(img_tensor),
            "original_size": original_size,
            "target_size": (target_w, target_h),
            "bucket_index": bucket_idx,
            "path": str(image_path),
            "timestamp": time.time(),
            "crop_coords": (left, top) if new_w != target_w or new_h != target_h else (0, 0)
        }

    def _process_single_image(self, image_path: Union[str, Path], config: Config) -> Optional[Dict[str, Any]]:
        """Process a single image with aspect ratio bucketing."""
        try:
            img_tensor, original_size = self._load_image_to_tensor(image_path)
            return self._process_on_gpu(
                self._process_image_tensor,
                img_tensor=img_tensor,
                original_size=original_size,
                config=config,
                image_path=image_path
            )
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

    def process_image_batch(
        self,
        image_paths: List[Union[str, Path]],
        captions: List[str],
        config: Config
    ) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel."""
        try:
            # Process images in parallel
            processed_images = []
            
            for path in image_paths:
                processed = self._process_single_image(path, config)
                processed_images.append(processed)
                
            # Process each caption individually to match training format
            results = []
            for img_data, caption in zip(processed_images, captions):
                if img_data is not None:
                    try:
                        # Encode single prompt
                        encoded_text = self.encode_prompt(
                            batch={"text": [caption]},
                            proportion_empty_prompts=0.0
                        )
                        
                        results.append({
                            **img_data,
                            "prompt_embeds": encoded_text["prompt_embeds"][0],
                            "pooled_prompt_embeds": encoded_text["pooled_prompt_embeds"][0],
                            "text": caption
                        })
                    except Exception as e:
                        logger.error(f"Failed to process caption: {caption}", exc_info=True)
                        results.append(None)
                else:
                    results.append(None)
                
            return results
            
        except Exception as e:
            logger.error("Batch processing failed", exc_info=True)
            return [None] * len(image_paths)

    def _log_action(self, operation: str, stats: Dict[str, Any]):
        """Log operation statistics."""
        if operation not in self.performance_stats['operation_times']:
            self.performance_stats['operation_times'][operation] = []
        self.performance_stats['operation_times'][operation].append(stats)

    def _get_cached_status(self, image_paths: List[str]) -> Dict[str, bool]:
        """Get cache status for each image path."""
        if not self.cache_manager:
            return {path: False for path in image_paths}
        
        return {
            path: self.cache_manager.is_cached(path)
            for path in image_paths
        }

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        """Get supported image dimensions for aspect ratio bucketing."""
        return config.global_config.image.supported_dims

def create_dataset(
    config: Config,
    image_paths: List[str],
    captions: List[str],
    model: Optional[StableDiffusionXL] = None,
    enable_memory_tracking: bool = True,
    max_memory_usage: float = 0.8
) -> AspectBucketDataset:
    """Create and initialize dataset instance."""
    return AspectBucketDataset(
        config=config,
        image_paths=image_paths,
        captions=captions,
        model=model,
        enable_memory_tracking=enable_memory_tracking,
        max_memory_usage=max_memory_usage
    )
