"""Enhanced progress tracking with performance analysis and historical prediction refinement."""
from typing import Optional, Any, Dict, List, Union, Callable, Iterator, TypeVar
from dataclasses import dataclass, field
from collections import deque
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .base import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

@dataclass
class HistoricalRun:
    """Data about a previous processing run."""
    total_items: int
    segment_times: Dict[str, List[float]]
    total_duration: float
    predicted_duration: float
    segment_rates: Dict[str, List[float]]
    bottlenecks: List[str]
    timestamp: float = field(default_factory=time.time)

    def prediction_accuracy(self) -> float:
        """Calculate prediction accuracy as percentage."""
        if self.total_duration == 0:
            return 0.0
        return (1 - abs(self.predicted_duration - self.total_duration) / self.total_duration) * 100

@dataclass
class PredictionModel:
    """Adaptive prediction model based on historical data."""
    history: List[HistoricalRun] = field(default_factory=list)
    max_history: int = 100
    weights: Dict[str, float] = field(default_factory=lambda: {"recent": 0.6, "similar": 0.3, "global": 0.1})
    
    def predict_duration(self, 
                        remaining_items: int, 
                        current_rates: Dict[str, float],
                        total_items: int) -> float:
        """Make prediction using weighted historical data."""
        if not self.history:
            # Fallback to simple prediction
            return self._simple_prediction(remaining_items, current_rates)
            
        predictions = []
        weights = []
        
        # Recent runs prediction
        if recent_pred := self._predict_from_recent(remaining_items, current_rates):
            predictions.append(recent_pred)
            weights.append(self.weights["recent"])
            
        # Similar size runs prediction
        if similar_pred := self._predict_from_similar_size(remaining_items, current_rates, total_items):
            predictions.append(similar_pred)
            weights.append(self.weights["similar"])
            
        # Global history prediction
        if global_pred := self._predict_from_global(remaining_items, current_rates):
            predictions.append(global_pred)
            weights.append(self.weights["global"])
            
        if not predictions:
            return self._simple_prediction(remaining_items, current_rates)
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        return float(np.average(predictions, weights=weights))
    
    def _simple_prediction(self, remaining_items: int, current_rates: Dict[str, float]) -> float:
        """Basic prediction without historical data."""
        if not current_rates:
            return float('inf')
        harmonic_mean = len(current_rates) / sum(1/r for r in current_rates.values())
        return remaining_items / harmonic_mean
    
    def _predict_from_recent(self, remaining_items: int, current_rates: Dict[str, float]) -> Optional[float]:
        """Predict based on recent runs."""
        if len(self.history) < 5:
            return None
            
        recent_runs = self.history[-5:]
        recent_accuracies = [run.prediction_accuracy() for run in recent_runs]
        weighted_rates = []
        
        for run, accuracy in zip(recent_runs, recent_accuracies):
            for segment, rates in run.segment_rates.items():
                if segment in current_rates:
                    weighted_rates.extend([r * (accuracy/100) for r in rates])
                    
        return remaining_items / np.mean(weighted_rates) if weighted_rates else None
    
    def _predict_from_similar_size(self, remaining_items: int, current_rates: Dict[str, float], total_items: int) -> Optional[float]:
        """Predict based on runs with similar total items."""
        similar_runs = [
            run for run in self.history 
            if 0.8 <= run.total_items / total_items <= 1.2
        ]
        if not similar_runs:
            return None
            
        similar_durations = [run.total_duration for run in similar_runs]
        return np.median(similar_durations) * (remaining_items / total_items)
    
    def _predict_from_global(self, remaining_items: int, current_rates: Dict[str, float]) -> Optional[float]:
        """Predict based on global history."""
        if not self.history:
            return None
            
        global_rates = []
        for run in self.history:
            for segment, rates in run.segment_rates.items():
                if segment in current_rates:
                    global_rates.extend(rates)
                    
        return remaining_items / np.median(global_rates) if global_rates else None
    
    def add_run(self, run: HistoricalRun) -> None:
        """Add new run to history and maintain size limit."""
        self.history.append(run)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def save(self, path: Path) -> None:
        """Save prediction model to file."""
        data = {
            "history": [vars(run) for run in self.history],
            "weights": self.weights,
            "max_history": self.max_history
        }
        path.write_text(json.dumps(data, indent=2))
        
    @classmethod
    def load(cls, path: Path) -> 'PredictionModel':
        """Load prediction model from file."""
        if not path.exists():
            return cls()
            
        data = json.loads(path.read_text())
        model = cls(
            max_history=data["max_history"],
            weights=data["weights"]
        )
        model.history = [HistoricalRun(**run) for run in data["history"]]
        return model

@dataclass
class SegmentStats:
    """Statistics for a specific processing segment."""
    durations: deque = field(default_factory=lambda: deque(maxlen=100))
    items_processed: deque = field(default_factory=lambda: deque(maxlen=100))
    rates: deque = field(default_factory=lambda: deque(maxlen=100))
    total_items: int = 0
    
    @property
    def avg_duration(self) -> float:
        return np.mean(self.durations) if self.durations else 0.0
    
    @property
    def avg_items_per_second(self) -> float:
        return np.mean(self.rates) if self.rates else 0.0
    
    @property
    def std_items_per_second(self) -> float:
        return np.std(self.rates) if len(self.rates) > 1 else 0.0

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
    bottleneck_threshold: float = 1.5
    history_path: Optional[Path] = None
    history_aware: bool = False  # Flag to enable/disable historical tracking

class ProgressTracker:
    """Enhanced progress tracking with optional historical performance analysis."""
    
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
        
        # Initialize tracking
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        self.processed = 0
        self._lock = threading.Lock()
        
        # Load prediction model only if history_aware is enabled
        self.prediction_model = (
            PredictionModel.load(self.config.history_path) 
            if self.config.history_aware and self.config.history_path 
            else None
        )
        
        # Performance tracking
        self.segments = {name: SegmentStats() for name in self.config.segment_names}
        self.bottleneck_history: List[str] = []
        self.current_prediction: float = 0.0
        
    def __enter__(self):
        """Initialize progress bar."""
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
        summary = self.close()
        if exc_type is None:
            self._log_final_stats(summary)

    def update(self, n: int = 1, segment: str = "main") -> None:
        """Update progress with segment-specific tracking."""
        with self._lock:
            current_time = time.perf_counter()
            duration = current_time - self.last_update_time
            
            # Update segment statistics
            if segment in self.segments:
                stats = self.segments[segment]
                stats.durations.append(duration)
                stats.items_processed.append(n)
                stats.rates.append(n/duration if duration > 0 else 0)
                stats.total_items += n
            
            # Update global progress
            self.processed += n
            self.pbar.update(n)
            
            # Update prediction and display
            self._update_display()
            
            # Detect bottlenecks
            self._analyze_bottlenecks()
            
            self.last_update_time = current_time

    def _predict_completion(self) -> float:
        """Predict remaining time using historical data if enabled, otherwise use simple prediction."""
        remaining = self.total - self.processed
        if remaining <= 0:
            return 0.0
            
        current_rates = {
            name: stats.avg_items_per_second
            for name, stats in self.segments.items()
            if stats.avg_items_per_second > 0
        }
        
        if self.config.history_aware and self.prediction_model:
            prediction = self.prediction_model.predict_duration(
                remaining, current_rates, self.total
            )
        else:
            # Simple prediction without historical data
            if not current_rates:
                return float('inf')
            harmonic_mean = len(current_rates) / sum(1/r for r in current_rates.values())
            prediction = remaining / harmonic_mean
            
        self.current_prediction = prediction
        return prediction

    def _analyze_bottlenecks(self) -> None:
        """Identify processing bottlenecks."""
        if len(self.segments) < 2:
            return
            
        avg_rates = {
            name: stats.avg_items_per_second
            for name, stats in self.segments.items()
            if stats.avg_items_per_second > 0
        }
        
        if not avg_rates:
            return
            
        median_rate = np.median(list(avg_rates.values()))
        bottlenecks = [
            name for name, rate in avg_rates.items()
            if rate < median_rate / self.config.bottleneck_threshold
        ]
        
        if bottlenecks and bottlenecks != self.bottleneck_history:
            self.bottleneck_history = bottlenecks
            self.logger.warning(
                "Performance bottleneck detected",
                extra={
                    'bottlenecks': bottlenecks,
                    'rates': {k: f"{v:.2f} items/s" for k, v in avg_rates.items()}
                }
            )

    def _update_display(self) -> None:
        """Update progress bar with current statistics."""
        eta = self._predict_completion()
        
        # Calculate rates for all segments
        rates = {
            name: stats.avg_items_per_second
            for name, stats in self.segments.items()
        }
        
        # Format display string
        rate_str = " | ".join(
            f"{name}: {rate:.2f} it/s"
            for name, rate in rates.items()
        )
        
        self.pbar.set_postfix_str(
            f"{rate_str} | ETA: {eta:.1f}s"
        )

    def close(self) -> Dict[str, Any]:
        """Close progress bar and optionally save historical data."""
        self.pbar.close()
        total_time = time.perf_counter() - self.start_time
        
        summary = {
            'total_time': total_time,
            'total_items': self.processed,
            'average_rate': self.processed / total_time,
            'segments': {
                name: {
                    'items_processed': stats.total_items,
                    'avg_rate': stats.avg_items_per_second,
                    'std_rate': stats.std_items_per_second
                }
                for name, stats in self.segments.items()
            },
            'bottlenecks': self.bottleneck_history
        }
        
        # Only track historical data if enabled
        if self.config.history_aware:
            run = HistoricalRun(
                total_items=self.total,
                segment_times={name: list(stats.durations) for name, stats in self.segments.items()},
                total_duration=total_time,
                predicted_duration=self.current_prediction,
                segment_rates={name: list(stats.rates) for name, stats in self.segments.items()},
                bottlenecks=self.bottleneck_history
            )
            
            if self.prediction_model:
                self.prediction_model.add_run(run)
                if self.config.history_path:
                    self.prediction_model.save(self.config.history_path)
                summary['prediction_accuracy'] = run.prediction_accuracy()
        
        return summary

    def _log_final_stats(self, summary: Dict[str, Any]) -> None:
        """Log final performance statistics."""
        self.logger.info(
            f"{self.config.desc} Complete",
            extra={
                'performance_summary': summary,
                **self.context
            }
        )

def track_parallel_progress(
    func: Callable[[T], Any],
    items: List[T],
    desc: str = "",
    max_workers: Optional[int] = None,
    segment_names: Optional[List[str]] = None,
    **kwargs
) -> List[Any]:
    """Execute tasks in parallel with performance tracking."""
    config = ProgressConfig(
        desc=desc,
        segment_names=segment_names or ["main"],
        **kwargs
    )
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