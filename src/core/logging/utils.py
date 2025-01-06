"""Logging utilities and progress tracking."""
from typing import Optional, Dict, Any, List, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from tqdm import tqdm

T = TypeVar('T')

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
    segment_names: List[str] = field(default_factory=lambda: ["main"])

class ProgressTracker:
    """Enhanced progress tracking."""
    def __init__(self, total: int, config: Optional[ProgressConfig] = None):
        self.total = total
        self.config = config or ProgressConfig()
        self.processed = 0
        self._lock = threading.Lock()
        self.start_time = time.perf_counter()
        
    def __enter__(self):
        self.pbar = tqdm(
            total=self.total,
            desc=self.config.desc,
            unit=self.config.unit
        )
        return self
        
    def __exit__(self, *args):
        self.pbar.close()
        
    def update(self, n: int = 1):
        with self._lock:
            self.processed += n
            self.pbar.update(n)

def track_parallel_progress(
    func: Callable[[T], Any],
    items: List[T],
    max_workers: Optional[int] = None,
    tracker: Optional[ProgressTracker] = None
) -> List[Any]:
    """Process items in parallel with progress tracking."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in items:
            future = executor.submit(func, item)
            future.add_done_callback(lambda _: tracker.update(1) if tracker else None)
            futures.append(future)
        results = [f.result() for f in futures]
    return results
