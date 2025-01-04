"""Distributed training utilities."""
import os
import random
from typing import Optional, Tuple, Dict, Any
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
import multiprocessing as mp
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DistributedError(Exception):
    """Base exception for distributed operations."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}

def find_free_port() -> int:
    """Find a free port using socket binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    except Exception as e:
        raise DistributedError("Failed to find free port", {"error": str(e)})

def get_unique_port() -> int:
    """Get a unique port in a range unlikely to be used by other programs."""
    try:
        base_port = 50000
        port_range = 5000
        rank_offset = int(os.environ.get("RANK", "0")) * 10
        random_offset = random.randint(0, port_range)
        return base_port + rank_offset + random_offset
    except Exception as e:
        raise DistributedError("Failed to generate unique port", {
            "rank": os.environ.get("RANK"),
            "error": str(e)
        })

def setup_training_env() -> None:
    """Setup training environment variables before any imports."""
    try:
        # Block accelerate initialization
        os.environ["ACCELERATE_DISABLE_RICH"] = "1"
        os.environ["ACCELERATE_USE_RICH"] = "0"
        os.environ["ACCELERATE_NO_AUTO_INIT"] = "1"
        
        # Set distributed variables with defaults
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        
        # Force PyTorch to use spawn method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
        logger.debug("Training environment initialized")
    except Exception as e:
        raise DistributedError("Failed to setup training environment", {
            "env_vars": dict(os.environ),
            "error": str(e)
        })

@contextmanager
def setup_environment():
    """Context manager for distributed training environment."""
    try:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if not dist.is_initialized():
                setup_distributed()
        yield
    finally:
        if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if dist.is_initialized():
                cleanup_distributed()
        from src.core.memory import torch_sync
        torch_sync()

def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
    max_port_tries: int = 5
) -> Tuple[int, int]:
    """Setup distributed training with robust port handling."""
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
    if world_size > 1:
        for attempt in range(max_port_tries):
            try:
                port = get_unique_port()
                os.environ["MASTER_PORT"] = str(port)
                
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=world_size,
                    rank=rank
                )
                
                logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}, port={port}, backend={backend}")
                break
                
            except Exception as e:
                if attempt == max_port_tries - 1:
                    raise DistributedError(
                        f"Failed to initialize distributed training after {max_port_tries} attempts",
                        {
                            "rank": rank,
                            "world_size": world_size,
                            "backend": backend,
                            "last_port": port,
                            "error": str(e)
                        }
                    )
                logger.warning(f"Port {port} failed, retrying...")
                continue
                
    return rank, world_size

def cleanup_distributed() -> None:
    """Cleanup distributed training resources safely."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.debug("Distributed process group destroyed")
    except Exception as e:
        raise DistributedError("Failed to cleanup distributed resources", {"error": str(e)})

def convert_model_to_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False
) -> DistributedDataParallel:
    """Convert model to DistributedDataParallel safely."""
    if not dist.is_initialized():
        logger.warning("Distributed training not initialized, returning unwrapped model")
        return model
    
    try:
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters
        )
    except Exception as e:
        raise DistributedError("Failed to convert model to DDP", {
            "model_type": type(model).__name__,
            "device_ids": device_ids,
            "error": str(e)
        })

def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size() -> int:
    """Get number of distributed processes."""
    return dist.get_world_size() if dist.is_initialized() else 1

def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary values across processes safely."""
    if not dist.is_initialized():
        return input_dict
        
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
        
    try:
        with torch.no_grad():
            names = []
            values = []
            
            for k, v in sorted(input_dict.items()):
                names.append(k)
                values.append(v)
                
            values = torch.stack(values, dim=0)
            dist.all_reduce(values)
            
            if average:
                values /= world_size
                
            return {k: v for k, v in zip(names, values)}
    except Exception as e:
        raise DistributedError("Failed to reduce dictionary", {
            "dict_keys": list(input_dict.keys()),
            "world_size": world_size,
            "error": str(e)
        })
