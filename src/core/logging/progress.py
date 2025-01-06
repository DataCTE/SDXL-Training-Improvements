"""Enhanced progress tracking with detailed logging integration."""
from typing import Optional, Any, Dict, List, Union, Callable, Iterator
import time
from dataclasses import dataclass
from tqdm import tqdm
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from .base import get_logger

logger = get_logger(__name__)

@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    desc: str = ""
    unit: str = "it"
    position: int = 0
    leave: bool = True
    dynamic_ncols: bool = True
    smoothing: float = 0.3
    disable: bool = False
    log_interval: int = 100
    log_level: int = logging.INFO
    show_memory: bool = True
    show_time_remaining: bool = True
    show_stats: bool = True

class ProgressTracker:
    """Enhanced progress tracking with logging integration."""
    
    def __init__(
        self, 
        total: int,
        config: Optional[ProgressConfig] = None,
        logger_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.total = total
        self.config = config or ProgressConfig()
        self.logger = get_logger(logger_name or __name__)
        self.context = context or {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._stats: Dict[str, Any] = {}
        
    def __enter__(self):
        """Initialize progress bar with configuration."""
        self.pbar = tqdm(
            total=self.total,
            desc=self.config.desc,
            unit=self.config.unit,
            position=self.config.position,
            leave=self.config.leave,
            dynamic_ncols=self.config.dynamic_ncols,
            smoothing=self.config.smoothing,
            disable=self.config.disable
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and log final statistics."""
        self.pbar.close()
        if exc_type is None:
            self._log_final_stats()
            
    def update(self, n: int = 1, stats: Optional[Dict[str, Any]] = None):
        """Update progress with optional statistics."""
        with self._lock:
            self.pbar.update(n)
            if stats:
                self._stats.update(stats)
                
            # Check if we should log progress
            current_time = time.time()
            if (current_time - self._last_log_time >= self.config.log_interval or 
                self.pbar.n >= self.total):
                self._log_progress()
                self._last_log_time = current_time
                
    def set_description(self, desc: str):
        """Update progress description."""
        with self._lock:
            self.pbar.set_description(desc)
            
    def set_postfix(self, **kwargs):
        """Update progress postfix."""
        with self._lock:
            self.pbar.set_postfix(**kwargs)
            
    def _log_progress(self):
        """Log current progress with statistics."""
        elapsed = time.time() - self._start_time
        progress = self.pbar.n / self.total
        
        log_data = {
            "progress": f"{progress:.1%}",
            "items_processed": self.pbar.n,
            "total_items": self.total,
            "elapsed": f"{elapsed:.1f}s"
        }
        
        if self.config.show_time_remaining and progress > 0:
            eta = elapsed / progress - elapsed
            log_data["eta"] = f"{eta:.1f}s"
            
        if self.config.show_stats and self._stats:
            log_data["stats"] = self._stats
            
        if self.config.show_memory:
            try:
                import torch
                if torch.cuda.is_available():
                    log_data["cuda_memory"] = {
                        "allocated": f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB",
                        "reserved": f"{torch.cuda.memory_reserved() / 1024**2:.1f}MB"
                    }
            except ImportError:
                pass
                
        self.logger.log(
            self.config.log_level,
            f"{self.config.desc} Progress",
            extra={"progress_data": log_data, **self.context}
        )
        
    def _log_final_stats(self):
        """Log final statistics after completion."""
        elapsed = time.time() - self._start_time
        
        final_stats = {
            "total_processed": self.pbar.n,
            "elapsed_time": f"{elapsed:.1f}s",
            "items_per_second": f"{self.pbar.n / elapsed:.1f}"
        }
        
        if self._stats:
            final_stats["statistics"] = self._stats
            
        self.logger.info(
            f"{self.config.desc} Complete",
            extra={"final_stats": final_stats, **self.context}
        )

def track_progress(
    iterable: Union[Iterator, List],
    desc: str = "",
    logger_name: Optional[str] = None,
    **kwargs
) -> Iterator:
    """Convenience function for progress tracking."""
    total = len(iterable) if hasattr(iterable, '__len__') else None
    config = ProgressConfig(desc=desc, **kwargs)
    
    with ProgressTracker(total, config, logger_name) as tracker:
        for item in iterable:
            yield item
            tracker.update(1)

def parallel_progress(
    func: Callable,
    items: List[Any],
    desc: str = "",
    max_workers: int = None,
    **kwargs
) -> List[Any]:
    """Execute tasks in parallel with progress tracking."""
    config = ProgressConfig(desc=desc, **kwargs)
    results = []
    
    with ProgressTracker(len(items), config) as tracker:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in items:
                future = executor.submit(func, item)
                future.add_done_callback(lambda _: tracker.update(1))
                futures.append(future)
                
            results = [f.result() for f in futures]
            
    return results 