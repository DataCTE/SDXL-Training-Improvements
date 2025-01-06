"""Enhanced logging utilities with detailed error tracking and formatting."""
import logging
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import torch
import colorama
from colorama import Fore, Style
from datetime import datetime
from logging import Logger, getLogger
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from .progress import ProgressTracker



# Initialize colorama for Windows support
colorama.init(autoreset=True)

class TensorLogger:
    """Specialized logger for tracking tensor shapes and statistics."""
    
    def __init__(self, parent_logger: logging.Logger, dump_on_error: bool = True):
        self.logger = parent_logger
        self._shape_logs = []
        self._lock = threading.Lock()
        self._dump_on_error = dump_on_error
        self._checkpoint_counter = 0
        
    def log_tensor(self, tensor: 'torch.Tensor', path: str, step: str) -> None:
        """Log tensor shape and statistics."""
        with self._lock:
            try:
                shape_info = {
                    'step': step,
                    'path': path,
                    'shape': tuple(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device),
                    'stats': self._compute_tensor_stats(tensor)
                }
                self._shape_logs.append(shape_info)
            except Exception as e:
                self.logger.warning(f"Failed to log tensor: {str(e)}", exc_info=True)
    
    def log_dict(self, data: Dict, path: str = "", step: str = "") -> None:
        """Recursively log dictionary of tensors."""
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                self.log_dict(value, current_path, step)
            elif hasattr(value, 'shape'):  # Tensor-like object
                self.log_tensor(value, current_path, step)
    
    def _compute_tensor_stats(self, tensor: 'torch.Tensor') -> Dict[str, float]:
        """Compute basic statistics for a tensor."""
        try:
            with torch.no_grad():
                tensor_cpu = tensor.detach().float().cpu()
                return {
                    'min': float(tensor_cpu.min().item()),
                    'max': float(tensor_cpu.max().item()),
                    'mean': float(tensor_cpu.mean().item()),
                    'std': float(tensor_cpu.std().item()) if tensor_cpu.numel() > 1 else 0.0,
                    'numel': tensor_cpu.numel()
                }
        except Exception:
            return {'error': 'Failed to compute statistics'}
    
    def log_checkpoint(self, name: str, tensors: Dict[str, Any] = None) -> None:
        """Log a named checkpoint with optional tensor state."""
        with self._lock:
            checkpoint = {
                'checkpoint_id': self._checkpoint_counter,
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'tensors': {},
                'memory': {
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }
            }
            
            if tensors:
                for key, tensor in tensors.items():
                    if isinstance(tensor, torch.Tensor):
                        checkpoint['tensors'][key] = {
                            'shape': tuple(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'stats': self._compute_tensor_stats(tensor)
                        }
                    elif isinstance(tensor, dict):
                        for subkey, subtensor in tensor.items():
                            if isinstance(subtensor, torch.Tensor):
                                full_key = f"{key}.{subkey}"
                                checkpoint['tensors'][full_key] = {
                                    'shape': tuple(subtensor.shape),
                                    'dtype': str(subtensor.dtype),
                                    'device': str(subtensor.device),
                                    'stats': self._compute_tensor_stats(subtensor)
                                }
            
            self._shape_logs.append(checkpoint)
            self._checkpoint_counter += 1
            
            # Log checkpoint immediately
            self.logger.debug(
                f"Checkpoint {name}:\n" + 
                "\n".join(f"  {k}: {v}" for k, v in checkpoint['tensors'].items())
            )

    def dump_shape_history(self, error_context: Optional[Dict] = None) -> None:
        """Dump complete shape history with improved formatting."""
        with self._lock:
            if not self._shape_logs:
                self.logger.warning("No shape history available")
                return

            history = ["=== Tensor Shape History ==="]
            
            if error_context:
                history.append("\nError Context:")
                history.extend(f"  {k}: {v}" for k, v in error_context.items())
            
            history.append("\nCheckpoint History:")
            for checkpoint in self._shape_logs:
                history.append(f"\nCheckpoint: {checkpoint['name']}")
                history.append(f"Timestamp: {checkpoint['timestamp']}")
                history.append(f"Memory Allocated: {checkpoint['memory']['allocated'] / 1024**2:.2f}MB")
                history.append(f"Memory Reserved: {checkpoint['memory']['reserved'] / 1024**2:.2f}MB")
                
                if checkpoint['tensors']:
                    history.append("Tensors:")
                    for tensor_name, tensor_info in checkpoint['tensors'].items():
                        history.append(f"  {tensor_name}:")
                        history.append(f"    Shape: {tensor_info['shape']}")
                        history.append(f"    Device: {tensor_info['device']}")
                        history.append(f"    Dtype: {tensor_info['dtype']}")
                        if 'stats' in tensor_info:
                            history.append(f"    Stats: {tensor_info['stats']}")

            self.logger.error("\n".join(history))
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Handle error by dumping shape history if enabled."""
        if self._dump_on_error:
            error_context = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            }
            self.dump_shape_history(error_context)

    def get_shape_history(self) -> List[Dict[str, Any]]:
        """Get the complete shape history."""
        with self._lock:
            return self._shape_logs.copy()
    
    def log_tensor_state(self, checkpoint_name: str) -> None:
        """Log the current state of all model tensors at a checkpoint."""
        with self._lock:
            try:
                memory_stats = {
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                    'max_allocated': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                }
                
                self._shape_logs.append({
                    'checkpoint': checkpoint_name,
                    'timestamp': datetime.now().isoformat(),
                    'memory_stats': memory_stats
                })
                
                self.logger.debug(
                    f"Logged tensor state at checkpoint: {checkpoint_name}",
                    extra={'memory_stats': memory_stats}
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to log tensor state: {str(e)}", exc_info=True)

    def clear_logs(self) -> None:
        """Clear the shape history."""
        with self._lock:
            self._shape_logs.clear()

# Global action history dict with thread safety
_action_history: Dict[str, Any] = {}
_history_lock = threading.Lock()

class EnhancedFormatter(logging.Formatter):
    """Custom formatter with colored output and context tracking."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.RED + Style.BRIGHT + Style.DIM
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and context."""
        # Add timestamp
        record.created_fmt = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Get base color
        color = self.COLORS.get(record.levelname, '')
        
        # Format basic message
        message = super().format(record)
        
        # Add context information if available
        if hasattr(record, 'context'):
            context_str = '\n'.join(f"  {k}: {v}" for k, v in record.context.items())
            message = f"{message}\nContext:\n{context_str}"
            
        # Add exception info with proper formatting
        if record.exc_info:
            if not message.endswith('\n'):
                message += '\n'
            message += self.formatException(record.exc_info)
            
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and getattr(record, 'stack_info', None):
            if not message.endswith('\n'):
                message += '\n'
            message += f"Stack trace:\n{record.stack_info}"
            
        # Apply color
        return f"{color}{message}{Style.RESET_ALL}"

def create_enhanced_logger(
    name: str,
    level: str = "INFO",
    console_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    file_level: Optional[str] = None,
    capture_warnings: bool = True,
    track_actions: bool = True
) -> logging.Logger:
    """Create an enhanced logger with console and file output.
    
    Args:
        name: Logger name
        level: Base logging level
        console_level: Console handler level (defaults to level)
        log_file: Optional log file path
        file_level: File handler level (defaults to level)
        capture_warnings: Whether to capture Python warnings
        track_actions: Whether to track actions in history
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, (console_level or level).upper()))
    console_handler.setFormatter(EnhancedFormatter(
        '%(created_fmt)s | %(levelname)s | %(name)s | %(message)s'
    ))
    logger.addHandler(console_handler)
    
    # File handler if path provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, (file_level or level).upper()))
        file_handler.setFormatter(logging.Formatter(
            '%(created_fmt)s.%(msecs)03d | %(levelname)s | %(name)s | '
            '%(processName)s:%(threadName)s | %(filename)s:%(lineno)d | '
            '%(funcName)s |\n%(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    # Configure warning capture
    if capture_warnings:
        logging.captureWarnings(True)
        
    def track_action(action: str, context: Optional[Dict] = None) -> None:
        """Track action in history with thread safety."""
        if track_actions:
            with _history_lock:
                timestamp = datetime.now().isoformat()
                _action_history[f"{timestamp}_{threading.get_ident()}"] = {
                    'action': action,
                    'context': context or {},
                    'logger': name,
                    'thread': threading.current_thread().name
                }
    
    # Add action tracking method
    logger.track_action = track_action
    
    return logger

def get_action_history() -> Dict[str, Any]:
    """Get copy of action history with thread safety."""
    with _history_lock:
        return _action_history.copy()

def clear_action_history() -> None:
    """Clear action history with thread safety."""
    with _history_lock:
        _action_history.clear()

def process_with_progress(
    items: List[Any],
    func: Callable,
    desc: str = "",
    logger: Optional[Logger] = None,
    max_workers: Optional[int] = None,
    segment_names: Optional[List[str]] = None,
    **kwargs
) -> List[Any]:
    """Process items in parallel with progress tracking."""
    if logger is None:
        logger = getLogger(__name__)
        
    with logger.create_progress_tracker(
        total=len(items),
        desc=desc,
        segment_names=segment_names,
        **kwargs
    ) as tracker:
        return track_parallel_progress(
            func=func,
            items=items,
            max_workers=max_workers,
            tracker=tracker
        )

class LogContext:
    """Context manager for scoped logging."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
        self.old_context = {}
        
    def __enter__(self):
        """Store previous context and set new one."""
        if hasattr(self.logger, "extra"):
            self.old_context = self.logger.extra
        self.logger.extra = self.context
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous context."""
        if self.old_context:
            self.logger.extra = self.old_context
        else:
            delattr(self.logger, "extra")

def log_errors(logger: logging.Logger):
    """Decorator to automatically log errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}",
                    exc_info=True,
                    extra={
                        'args': args,
                        'kwargs': kwargs,
                        'error': str(e)
                    }
                )
                raise
        return wrapper
    return decorator

def track_parallel_progress(
    func: Callable,
    items: List[Any],
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
