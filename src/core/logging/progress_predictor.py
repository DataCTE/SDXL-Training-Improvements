"""Progress prediction utilities for accurate ETA estimation."""
import time
from typing import Optional, Dict
from collections import deque
import numpy as np

class ProgressPredictor:
    """Smart progress predictor using multi-window moving averages."""
    
    def __init__(self):
        """Initialize progress predictor."""
        # Per-item timing
        self._last_update = None
        self._current_time = 0.0
        
        # Moving averages for different windows
        self._times_short = deque(maxlen=10)  # 10-item window
        self._times_long = deque(maxlen=100)  # 100-item window
        
        # Total progress tracking
        self._total_items = 0
        self._completed_items = 0
        self._start_time = None
        
    def start(self, total_items: int) -> None:
        """Start progress tracking.
        
        Args:
            total_items: Total number of items to process
        """
        self._total_items = total_items
        self._completed_items = 0
        self._start_time = time.time()
        self._last_update = self._start_time
        
    def update(self, items: int = 1) -> Dict[str, float]:
        """Update progress and get timing predictions.
        
        Args:
            items: Number of items completed in this update
            
        Returns:
            Dictionary with timing predictions:
            - current_time: Time per item for latest update
            - eta_seconds: Estimated seconds remaining
        """
        current_time = time.time()
        
        if self._last_update is None:
            self._last_update = current_time
            return {"current_time": 0.0, "eta_seconds": 0.0}
            
        # Calculate time per item for this update
        elapsed = current_time - self._last_update
        time_per_item = elapsed / items
        self._current_time = time_per_item
        
        # Update moving averages
        self._times_short.append(time_per_item)
        self._times_long.append(time_per_item)
        
        # Update progress
        self._completed_items += items
        self._last_update = current_time
        
        # Calculate progress percentage
        progress = (self._completed_items / self._total_items) * 100 if self._total_items > 0 else 0
        
        # Calculate predictions
        remaining_items = self._total_items - self._completed_items
        if remaining_items <= 0:
            return {"current_time": time_per_item, "eta_seconds": 0.0}
            
        # Use weighted average of different windows
        if len(self._times_long) >= 10:
            # We have enough data for both windows
            avg_short = np.mean(self._times_short)
            avg_long = np.mean(self._times_long)
            # Weight recent times more heavily
            predicted_per_item = 0.7 * avg_short + 0.3 * avg_long
        elif len(self._times_short) >= 3:
            # Only use short window
            predicted_per_item = np.mean(self._times_short)
        else:
            # Use current time as best guess
            predicted_per_item = time_per_item
            
        eta_seconds = remaining_items * predicted_per_item
        
        return {
            "current_time": time_per_item,
            "eta_seconds": eta_seconds,
            "progress": progress,
            "completed": self._completed_items,
            "total": self._total_items
        }
        
    def get_progress(self) -> Dict[str, float]:
        """Get current progress statistics.
        
        Returns:
            Dictionary with progress info:
            - progress: Progress as fraction (0-1)
            - elapsed_seconds: Total elapsed time
            - current_time: Current time per item
            - eta_seconds: Estimated time remaining
        """
        if self._start_time is None:
            return {
                "progress": 0.0,
                "elapsed_seconds": 0.0,
                "current_time": 0.0,
                "eta_seconds": 0.0
            }
            
        current = time.time()
        elapsed = current - self._start_time
        progress = self._completed_items / self._total_items if self._total_items > 0 else 0.0
        
        # Get timing predictions
        predictions = self.update(0)  # Get predictions without updating progress
        
        return {
            "progress": progress,
            "elapsed_seconds": elapsed,
            "current_time": predictions["current_time"],
            "eta_seconds": predictions["eta_seconds"]
        }
        
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to human readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (e.g. "2h 3m 45s")
        """
        if seconds < 0:
            return "Unknown"
            
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)
