"""High-performance preprocessing pipeline with extreme speedups."""
import time
import torch
import random
import logging
from contextlib import contextmanager
from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.models.encoders import CLIPEncoder
from src.data.utils.paths import convert_windows_path, is_windows_path, convert_path_list

logger = get_logger(__name__)
import torch.backends.cudnn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
from dataclasses import dataclass
from src.data.preprocessing.cache_manager import CacheManager
from src.core.memory.tensor import create_stream_context
import numpy as np
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
        num_gpu_workers: int = 1,
        num_cpu_workers: int = 4,
        num_io_workers: int = 2,
        prefetch_factor: int = 2,
        use_pinned_memory: bool = True,
        enable_memory_tracking: bool = True,
        stream_timeout: float = 10.0
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
            
            self.device_id = 0  # Always use first GPU
            self.device = torch.device(f'cuda:{self.device_id}')
            torch.cuda.set_device(self.device_id)
        else:
            self.device_id = None
            self.device = torch.device('cpu')
            
        # Core components
        self.config = config
        self.model = model
        self.vae_encoder = model.vae_encoder
        self.text_encoders = model.text_encoders
        self.tokenizers = model.tokenizers
        
        self.cache_manager = cache_manager
        self.is_train = is_train
        
        # Worker configuration
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        
        # Stream management
        self._streams = {}
        self.prefetch_factor = prefetch_factor
        self.use_pinned_memory = use_pinned_memory
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

    def _process_image(self, img_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process single image using VAE encoder directly."""
        try:
            # Convert Windows path if needed
            img_path = convert_windows_path(img_path) if is_windows_path(img_path) else Path(img_path)
            img = Image.open(img_path).convert('RGB')
            
            if not self.validate_image_size(img.size):
                logger.warning(f"Image {img_path} failed size validation")
                return None

            resized_img, bucket_idx = self.resize_to_bucket(img)
            
            tensor = torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            
            with torch.cuda.amp.autocast():
                with self.track_memory_usage("vae_encoding"):
                    # Direct VAE encoding implementation
                    with torch.cuda.amp.autocast(enabled=False), torch.no_grad():
                        model_input = self.model.vae_encoder.vae.encode(tensor).latent_dist.sample()
                        model_input = model_input * self.model.vae_encoder.vae.config.scaling_factor
                        model_input = model_input.cpu()
                    
                    metadata = {
                        "original_size": img.size,
                        "bucket_size": self.buckets[bucket_idx],
                        "bucket_index": bucket_idx,
                        "path": str(img_path),
                        "timestamp": time.time(),
                        "scaling_factor": self.model.vae_encoder.vae.config.scaling_factor,
                        "input_shape": tuple(tensor.shape),
                        "output_shape": tuple(model_input.shape)
                    }
                    
                    return {
                        "model_input": model_input,
                        "metadata": metadata
                    }

        except Exception as e:
            self.stats.failed += 1
            logger.error(f"Failed to process {img_path}: {str(e)}")
            self.performance_stats['errors'].append({
                'path': str(img_path),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': time.time()
            })
            return None

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
        try:
            # Load and validate image
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            
            # Calculate aspect ratio
            aspect_ratio = w / h
            
            # Check if aspect ratio is within bounds
            if aspect_ratio > config.global_config.image.max_aspect_ratio or \
               aspect_ratio < (1 / config.global_config.image.max_aspect_ratio):
                logger.warning(f"Skipping image {image_path} - aspect ratio {aspect_ratio:.2f} exceeds bounds")
                return None
            
            # Find best matching bucket dimensions
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
            
            # Calculate resize dimensions while preserving aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop if needed
            if new_w != target_w or new_h != target_h:
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                right = left + target_w
                bottom = top + target_h
                img = img.crop((left, top, right, bottom))
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW
            
            # Ensure tensor is contiguous
            img_tensor = img_tensor.contiguous()
            if not img_tensor.is_floating_point():
                img_tensor = img_tensor.float()
            
            # Return in the format expected by the pipeline
            return {
                "model_input": img_tensor,
                "original_size": (w, h),
                "bucket_size": (target_w, target_h),
                "bucket_index": bucket_idx,
                "path": str(image_path),
                "timestamp": time.time(),
                "crop_coords": (left, top) if new_w != target_w or new_h != target_h else (0, 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None
