"""LoRA and additional embedding wrapper modules."""
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

class LoRAModuleWrapper(nn.Module):
    """Wrapper for LoRA (Low-Rank Adaptation) training."""
    
    def __init__(
        self,
        base_module: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        """Initialize LoRA wrapper."""
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Initialize LoRA matrices
        if isinstance(base_module, nn.Linear):
            in_features = base_module.in_features
            out_features = base_module.out_features
        elif isinstance(base_module, nn.Conv2d):
            in_features = base_module.in_channels
            out_features = base_module.out_channels
        else:
            raise ValueError(f"Unsupported module type: {type(base_module)}")
            
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # Initialize weights
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base and LoRA outputs."""
        base_output = self.base_module(x)
        
        # Apply LoRA path
        lora_output = self.dropout_layer(x)
        lora_output = self.lora_down(lora_output)
        lora_output = self.lora_up(lora_output)
        
        # Scale and combine
        return base_output + self.alpha * lora_output

class AdditionalEmbeddingWrapper(nn.Module):
    """Wrapper for training additional embeddings."""
    
    def __init__(
        self,
        base_embeddings: nn.Module,
        num_additional_tokens: int,
        embedding_dim: int
    ):
        """Initialize embedding wrapper."""
        super().__init__()
        self.base_embeddings = base_embeddings
        
        # Create additional embeddings
        self.additional_embeddings = nn.Embedding(
            num_additional_tokens,
            embedding_dim
        )
        
        # Initialize weights
        with torch.no_grad():
            std = self.base_embeddings.weight.std()
            nn.init.normal_(self.additional_embeddings.weight, std=std)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        additional_token_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass combining base and additional embeddings."""
        base_embeds = self.base_embeddings(input_ids, **kwargs)
        
        if additional_token_ids is not None:
            additional_embeds = self.additional_embeddings(additional_token_ids)
            
            # Replace embeddings for additional tokens
            mask = (additional_token_ids != -1).unsqueeze(-1)
            base_embeds = torch.where(mask, additional_embeds, base_embeds)
            
        return base_embeds
