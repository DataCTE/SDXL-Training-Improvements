"""Text embedding processor with support for custom embeddings."""
from typing import Dict, List, Optional, Union, Any
import torch
from torch import Tensor
import logging
from src.core.logging.logging import setup_logging
from src.models.embeddings import BaseModelEmbedding

logger = setup_logging(__name__)

class TextEmbeddingProcessor:
    """Handles text embedding processing with custom embedding support."""
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
        enable_memory_tracking: bool = True
    ):
        """Initialize embedding processor.
        
        Args:
            device: Target device
            dtype: Model dtype
            enable_memory_tracking: Whether to track memory usage
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_memory_tracking = enable_memory_tracking
        self.memory_stats = {
            'peak_allocated': 0,
            'current_allocated': 0
        }
        
        logger.info("Text embedding processor initialized", extra={
            'device': str(self.device),
            'dtype': str(self.dtype)
        })

    def process_embeddings(
        self,
        prompt: str,
        additional_embeddings: Optional[List[BaseModelEmbedding]] = None,
        base_embedding: Optional[BaseModelEmbedding] = None
    ) -> Dict[str, Any]:
        """Process text with optional custom embeddings.
        
        Args:
            prompt: Input text prompt
            additional_embeddings: Optional list of additional embeddings
            base_embedding: Optional base embedding
            
        Returns:
            Dictionary containing processed embeddings and metadata
        """
        try:
            # Track memory if enabled
            if self.enable_memory_tracking and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()

            # Start with original prompt
            modified_prompt = prompt

            # Add additional embeddings
            if additional_embeddings:
                for embedding in additional_embeddings:
                    if not isinstance(embedding, BaseModelEmbedding):
                        logger.warning(f"Invalid additional embedding type: {type(embedding)}")
                        continue
                    modified_prompt = f"{embedding.placeholder} {modified_prompt}"

            # Add base embedding
            if base_embedding and isinstance(base_embedding, BaseModelEmbedding):
                modified_prompt = f"{base_embedding.placeholder} {modified_prompt}"

            # Clean up whitespace
            modified_prompt = " ".join(modified_prompt.split())

            # Prepare metadata
            metadata = {
                "original_prompt": prompt,
                "modified_prompt": modified_prompt,
                "has_additional_embeddings": bool(additional_embeddings),
                "has_base_embedding": bool(base_embedding),
                "num_additional_embeddings": len(additional_embeddings) if additional_embeddings else 0
            }

            # Track memory usage
            if self.enable_memory_tracking and torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                self.memory_stats.update({
                    'peak_allocated': max(self.memory_stats['peak_allocated'], peak_memory),
                    'current_allocated': current_memory,
                    'memory_change': current_memory - start_memory
                })

            return {
                "processed_text": modified_prompt,
                "metadata": metadata
            }

        except Exception as e:
            logger.error("Failed to process embeddings", extra={
                'error': str(e),
                'prompt': prompt,
                'stack_trace': True
            })
            raise

    def combine_embeddings(
        self,
        text_embeddings: torch.Tensor,
        additional_embeddings: Optional[List[BaseModelEmbedding]] = None,
        base_embedding: Optional[BaseModelEmbedding] = None
    ) -> torch.Tensor:
        """Combine text embeddings with custom embeddings.
        
        Args:
            text_embeddings: Base text embeddings
            additional_embeddings: Optional list of additional embeddings
            base_embedding: Optional base embedding
            
        Returns:
            Combined embedding tensor
        """
        try:
            # Start with base text embeddings
            combined = text_embeddings

            # Add additional embeddings
            if additional_embeddings:
                for embedding in additional_embeddings:
                    if not isinstance(embedding, BaseModelEmbedding):
                        continue
                    # Move embedding to correct device/dtype
                    vector = embedding.text_encoder_1_vector.to(
                        device=self.device,
                        dtype=self.dtype
                    )
                    combined = torch.cat([combined, vector], dim=-1)

            # Add base embedding
            if base_embedding and isinstance(base_embedding, BaseModelEmbedding):
                vector = base_embedding.text_encoder_1_vector.to(
                    device=self.device,
                    dtype=self.dtype
                )
                combined = torch.cat([combined, vector], dim=-1)

            return combined

        except Exception as e:
            logger.error("Failed to combine embeddings", extra={
                'error': str(e),
                'embedding_shape': tuple(text_embeddings.shape),
                'stack_trace': True
            })
            raise

    def validate_embeddings(
        self,
        embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> bool:
        """Validate embedding tensors.
        
        Args:
            embeddings: Tensor or dict of tensors to validate
            
        Returns:
            bool indicating if embeddings are valid
        """
        try:
            if isinstance(embeddings, dict):
                tensors = embeddings.values()
            else:
                tensors = [embeddings]

            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor):
                    return False
                    
                if torch.isnan(tensor).any():
                    logger.warning("NaN values detected in embeddings")
                    return False
                    
                if torch.isinf(tensor).any():
                    logger.warning("Infinite values detected in embeddings")
                    return False

            return True

        except Exception as e:
            logger.error(f"Embedding validation failed: {str(e)}")
            return False
