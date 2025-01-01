"""High-performance preprocessing pipeline with extreme speedups."""
import time
import torch
import random
import logging
from contextlib import contextmanager
from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.models.encoders import CLIPEncoder
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_paths

logger = get_logger(__name__)
import torch.backends.cudnn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
from dataclasses import dataclass
from src.data.preprocessing.cache_manager import CacheManager
from src.core.memory.tensor import create_stream_context
import numpy as np
import torch.nn.functional as F
from src.data.config import Config

class ProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass

@dataclass
class PipelineStats:
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    gpu_oom_events: int = 0
    stream_sync_failures: int = 0
    dtype_conversion_errors: int = 0

class PreprocessingPipeline:
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXL,
        cache_manager: Optional[CacheManager] = None,
        is_train: bool = True,
        enable_memory_tracking: bool = True,
        stream_timeout: float = 10.0,
        device: Optional[torch.device] = None,
        device_id: Optional[int] = None
    ):
        """Initialize preprocessing pipeline."""
        # Performance tracking
        self.performance_stats = {
            'operation_times': {},
            'memory_usage': {},
            'errors': []
        }
        self.logger = logger
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
        self.config = config
        self.model = model
        self.vae_encoder = model.vae_encoder
        self.text_encoders = model.text_encoders
        self.tokenizers = model.tokenizers
        self.cache_manager = cache_manager
        self.is_train = is_train
        
        # Stream management
        self._streams = {}
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_timeout = stream_timeout
        self.stats = PipelineStats()

        # Initialize buckets
        self.buckets = self.get_aspect_buckets(config)
        self.bucket_indices = []

    def get_aspect_buckets(self, config: Config) -> List[Tuple[int, int]]:
        return config.global_config.image.supported_dims

    def _assign_single_bucket(
        self,
        img: Union[str, Path, Image.Image],
        max_aspect_ratio: Optional[float] = None
    ) -> Tuple[int, Tuple[int, int]]:
        """Assign image to optimal bucket based on aspect ratio and size."""
        try:
            if isinstance(img, (str, Path)):
                # Convert Windows path if needed
                img_path = convert_windows_path(img) if is_windows_path(img) else Path(img)
                img = Image.open(img_path).convert('RGB')
                
            w, h = img.size
            aspect_ratio = w / h
            
            max_ar = max_aspect_ratio or self.config.global_config.image.max_aspect_ratio
            
            # Find exact matching bucket first
            for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
                bucket_ratio = bucket_w / bucket_h
                if abs(aspect_ratio - bucket_ratio) < 0.01:
                    return idx, (bucket_h, bucket_w)
            
            # If no exact match, find closest within max aspect ratio
            closest_idx = None
            min_diff = float('inf')
            
            for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
                bucket_ratio = bucket_w / bucket_h
                if abs(bucket_ratio - aspect_ratio) < min_diff:
                    if abs(bucket_ratio - aspect_ratio) <= max_ar:
                        closest_idx = idx
                        min_diff = abs(bucket_ratio - aspect_ratio)
            
            if closest_idx is not None:
                return closest_idx, self.buckets[closest_idx]
                
            # If no suitable bucket found
            raise ValueError(f"No suitable bucket found for aspect ratio {aspect_ratio}")
            
        except Exception as e:
            logger.error(f"Bucket assignment failed: {str(e)}")
            raise

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
        """Context manager for tracking memory usage during operations."""
        try:
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            yield
            
        finally:
            duration = time.time() - start_time
            memory_stats = {}
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_stats.update({
                    'start_memory': start_memory,
                    'end_memory': end_memory,
                    'peak_memory': peak_memory,
                    'memory_change': end_memory - start_memory
                })
                
            self._log_action(operation, {
                'duration': duration,
                'memory_stats': memory_stats
            })

    def _get_stream(self):
        """Get a CUDA stream for the current thread."""
        import threading
        thread_id = threading.get_ident()
        if thread_id not in self._streams and torch.cuda.is_available():
            self._streams[thread_id] = torch.cuda.Stream()
        return self._streams.get(thread_id)

    def _process_on_gpu(self, func, *args, **kwargs):
        """Execute function on GPU with proper stream management."""
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
            
        try:
            with torch.cuda.device(self.device_id):
                stream = self._get_stream()
                with create_stream_context(stream):
                    result = func(*args, **kwargs)
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

    def _log_action(self, operation: str, stats: Dict[str, Any]):
        """Log operation statistics."""
        if operation not in self.performance_stats['operation_times']:
            self.performance_stats['operation_times'][operation] = []
        self.performance_stats['operation_times'][operation].append(stats)

    def _assign_bucket_indices(self, image_paths: List[str]) -> List[int]:
        """Assign bucket indices for a list of image paths."""
        bucket_indices = []
        
        for img_path in image_paths:
            try:
                # Use existing single bucket assignment
                bucket_idx, _ = self._assign_single_bucket(img_path)
                bucket_indices.append(bucket_idx)
            except Exception as e:
                logger.error(f"Failed to assign bucket for {img_path}: {str(e)}")
                # Assign to default bucket (square) in case of error
                bucket_indices.append(0)
                
        return bucket_indices

    def process_image_batch(
        self,
        image_paths: List[Union[str, Path]],
        captions: List[str],
        config: Config
    ) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of images in parallel but return individual results.
        
        Args:
            image_paths: List of paths to images
            captions: List of captions for the images
            config: Configuration object
            
        Returns:
            List of individual processed items, each can be None if processing failed
        """
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
                            "prompt_embeds": encoded_text["prompt_embeds"][0],  # Take first item
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

    def encode_prompt_batch(
        self,
        batch: Dict[str, List[str]],
        proportion_empty_prompts: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of prompts using text encoders."""
        try:
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = [], []
                
                # Process in smaller sub-batches if needed
                sub_batch_size = 32
                texts = batch["text"]
                
                for i in range(0, len(texts), sub_batch_size):
                    sub_texts = texts[i:i + sub_batch_size]
                    
                    # Add empty prompts if requested
                    if proportion_empty_prompts > 0:
                        num_empty = int(len(sub_texts) * proportion_empty_prompts)
                        if num_empty > 0:
                            sub_texts[:num_empty] = [""] * num_empty
                    
                    # Encode sub-batch
                    sub_embeds = self.text_encoders(sub_texts)
                    prompt_embeds.append(sub_embeds["prompt_embeds"])
                    pooled_prompt_embeds.append(sub_embeds["pooled_prompt_embeds"])
                
                # Concatenate results
                return {
                    "prompt_embeds": torch.cat(prompt_embeds, dim=0),
                    "pooled_prompt_embeds": torch.cat(pooled_prompt_embeds, dim=0)
                }
                
        except Exception as e:
            logger.error("Failed to encode prompt batch", exc_info=True)
            raise

    def _process_single_image(self, image_path: Union[str, Path], config: Config) -> Optional[Dict[str, Any]]:
        """Process a single image with aspect ratio bucketing.
        
        Args:
            image_path: Path to image file
            config: Configuration object
            
        Returns:
            Optional[Dict[str, Any]]: Processed image data or None if processing fails
        """
        self._check_initialized()
        try:
            # Load and validate image
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            
            # Early conversion to tensor and move to GPU
            img_tensor = torch.from_numpy(np.array(img)).float().to(self.device) / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
            
            # Process on GPU
            return self._process_on_gpu(
                self._process_image_tensor,
                img_tensor=img_tensor,
                original_size=(w, h),
                config=config,
                image_path=image_path
            )
        
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None

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
        
        # Check aspect ratio
        if aspect_ratio > config.global_config.image.max_aspect_ratio or \
           aspect_ratio < (1 / config.global_config.image.max_aspect_ratio):
            return None
        
        # Find best bucket and resize/crop
        target_w, target_h = self.get_aspect_buckets(config)[0]
        min_diff = float('inf')
        bucket_idx = 0
        
        for idx, (bucket_w, bucket_h) in enumerate(self.get_aspect_buckets(config)):
            bucket_ratio = bucket_w / bucket_h
            diff = abs(aspect_ratio - bucket_ratio)
            if diff < min_diff:
                min_diff = diff
                target_w, target_h = bucket_w, bucket_h
                bucket_idx = idx
        
        # Resize on GPU
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Center crop on GPU
        if new_w != target_w or new_h != target_h:
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img_tensor = img_tensor[:, top:top + target_h, left:left + target_w]
        
        # After resizing and cropping, encode with VAE
        with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0)
            latents = self.model.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor
            latents = latents.cpu()

        return {
            "pixel_values": latents,  # Now this contains VAE latents
            "original_size": original_size,
            "target_size": (target_w, target_h),
            "bucket_index": bucket_idx,
            "path": str(image_path),
            "timestamp": time.time(),
            "crop_coords": (left, top) if new_w != target_w or new_h != target_h else (0, 0)
        }

    def _get_cached_status(self, image_paths: List[str]) -> Dict[str, bool]:
        """Get cache status for each image path."""
        if not self.cache_manager:
            return {path: False for path in image_paths}
            
        return {
            path: self.cache_manager.is_cached(path)
            for path in image_paths
        }

    def _check_initialized(self) -> None:
        """Check if pipeline is properly initialized."""
        if not all([self.model, self.vae_encoder, self.text_encoders, self.tokenizers]):
            raise RuntimeError(
                "Pipeline not initialized. Call initialize_worker() first."
            )
