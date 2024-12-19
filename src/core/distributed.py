"""Distributed training utilities."""
import logging
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)

def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl"
) -> Tuple[int, int]:
    """Setup distributed training.
    
    Args:
        rank: Process rank. If None, get from env
        world_size: Total number of processes. If None, get from env
        backend: PyTorch distributed backend
        
    Returns:
        Tuple of (rank, world_size)
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        logger.info(f"Initialized distributed training with backend {backend}")
        logger.info(f"Process rank: {rank}, World size: {world_size}")
        
    return rank, world_size

def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def convert_model_to_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False
) -> DistributedDataParallel:
    """Convert model to DistributedDataParallel.
    
    Args:
        model: Model to convert
        device_ids: List of GPU device IDs
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        logger.warning("Distributed training not initialized, returning unwrapped model")
        return model
        
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters
    )

def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size() -> int:
    """Get number of distributed processes."""
    return dist.get_world_size() if dist.is_initialized() else 1

def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary values across processes.
    
    Args:
        input_dict: Dictionary to reduce
        average: Whether to average or sum
        
    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict
        
    world_size = dist.get_world_size()
    
    if world_size < 2:
        return input_dict
        
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
            
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict
