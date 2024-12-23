"""High-performance preprocessing pipeline with extreme speedups."""
import logging
import time
import torch
import torch.backends.cudnn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)

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
        config,
        latent_preprocessor=None,
        cache_manager=None,
        is_train=True,
        num_gpu_workers=1,
        num_cpu_workers=4,
        num_io_workers=2,
        prefetch_factor=2,
        device_ids=None,
        use_pinned_memory=True,
        enable_memory_tracking=True,
        stream_timeout=10.0
    ):
        # Basic initialization
        self.config = config
        self.latent_preprocessor = latent_preprocessor
        self.cache_manager = cache_manager
        self.is_train = is_train
        self.num_gpu_workers = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.num_io_workers = num_io_workers
        self.prefetch_factor = prefetch_factor
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.use_pinned_memory = use_pinned_memory
        self.enable_memory_tracking = enable_memory_tracking
        self.stream_timeout = stream_timeout
        self.stats = PipelineStats()
        self.input_queue = Queue(maxsize=prefetch_factor * num_gpu_workers)
        self.output_queue = Queue(maxsize=prefetch_factor * num_gpu_workers)
        self._init_pools()
        if hasattr(torch, "compile"):
            self.precompute_latents = torch.compile(self.precompute_latents, mode="reduce-overhead", fullgraph=True)
            self._process_image = torch.compile(self._process_image, mode="reduce-overhead", fullgraph=True)

    def _init_pools(self):
        self.gpu_pool = (ThreadPoolExecutor(max_workers=self.num_gpu_workers) if torch.cuda.is_available() else None)
        self.cpu_pool = ProcessPoolExecutor(max_workers=self.num_cpu_workers)
        self.io_pool = ThreadPoolExecutor(max_workers=self.num_io_workers)

    def precompute_latents(self, image_paths, captions, latent_preprocessor, batch_size=1, proportion_empty_prompts=0.0):
        if not latent_preprocessor or not self.cache_manager or not self.is_train:
            return
        logger.info(f"Precomputing {len(image_paths)} latents")
        to_process = []
        cached = set(self.cache_manager.cache_index["files"].keys()) if self.cache_manager else set()
        for path in image_paths:
            if path not in cached: to_process.append(path)
            else: self.stats.cache_hits += 1
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i+batch_size]
            batch_caps = [captions[image_paths.index(p)] for p in batch]
            for img_path, cap in zip(batch, batch_caps):
                try:
                    processed = self._process_image(img_path)
                    if processed:
                        self.cache_manager.save_preprocessed_data(
                            latent_data=processed["latent"],
                            text_embeddings=processed.get("text_embeddings"),
                            metadata=processed.get("metadata", {}),
                            file_path=img_path
                        )
                        self.stats.cache_misses += 1
                        self.stats.successful += 1
                except Exception as e:
                    self.stats.failed += 1
                    logger.warning(e)

    def _process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            metadata = {"original_size": img.size, "path": str(img_path), "timestamp": time.time()}
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            if torch.cuda.is_available():
                tensor = tensor.cuda(non_blocking=True).to(dtype=torch.float16)
            if self.latent_preprocessor:
                latent = self.latent_preprocessor.encode_images(tensor)
            else:
                latent = tensor
            return {"latent": latent, "metadata": metadata}
        except Exception as e:
            self.stats.failed += 1
            logger.warning(f"Failed to process {img_path}: {e}")
            return None
