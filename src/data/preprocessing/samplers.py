"""Samplers for efficient batch construction."""
import random
from typing import Dict, List, Tuple, Any
from src.core.logging import UnifiedLogger, LogConfig

logger = UnifiedLogger(LogConfig(name=__name__))

class BucketBatchSampler:
    """Samples batches ensuring all items in a batch are from the same bucket."""
    
    def __init__(self, bucket_indices: Dict[Tuple[int, ...], List[int]], 
                 batch_size: int, 
                 drop_last: bool = True, 
                 shuffle: bool = True):
        """Initialize bucket sampler.
        
        Args:
            bucket_indices: Dictionary mapping bucket shapes to lists of dataset indices
            batch_size: Number of items per batch
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle batches
        """
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Create batches for each bucket
        self.batches = []
        
        # Log bucket statistics
        logger.info("Creating bucket batches:")
        for bucket_shape, indices in bucket_indices.items():
            # Skip buckets with too few samples if dropping last
            if len(indices) < batch_size and drop_last:
                logger.info(f"  Skipping bucket {bucket_shape} with only {len(indices)} samples")
                continue
            
            # Create batches for this bucket
            bucket_batches = [
                indices[i:i + batch_size] 
                for i in range(0, len(indices), batch_size)
            ]
            
            # Drop last incomplete batch if needed
            if drop_last and len(bucket_batches[-1]) < batch_size:
                bucket_batches = bucket_batches[:-1]
            
            self.batches.extend(bucket_batches)
            logger.info(f"  Bucket {bucket_shape}: {len(bucket_batches)} batches")
        
        if not self.batches:
            raise ValueError("No valid batches created - check bucket sizes and batch size")
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)