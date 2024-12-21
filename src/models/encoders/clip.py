"""CLIP model encoding utilities."""
from typing import Optional, Tuple

import torch
from torch import Tensor
from transformers import CLIPTextModel

import logging

logger = logging.getLogger(__name__)

def encode_clip(
    text_encoder: CLIPTextModel,
    tokens: Tensor,
    default_layer: int = -2,
    layer_skip: int = 0,
    text_encoder_output: Optional[Tensor] = None,
    add_pooled_output: bool = False,
    pooled_text_encoder_output: Optional[Tensor] = None,
    use_attention_mask: bool = False,
    add_layer_norm: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Encode text tokens using CLIP text encoder.
    
    Returns:
        Tuple of (encoder output tensor, optional pooled output tensor)
    """
    if text_encoder_output is None:
        # Create attention mask if needed
        attention_mask = None
        if use_attention_mask:
            attention_mask = tokens.ne(text_encoder.config.pad_token_id).long()
            
        # Get encoder outputs
        outputs = text_encoder(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states from specified layer
        target_layer = default_layer - layer_skip
        text_encoder_output = outputs.hidden_states[target_layer]
        
        # Add layer norm if configured
        if add_layer_norm and hasattr(text_encoder, "final_layer_norm"):
            text_encoder_output = text_encoder.final_layer_norm(text_encoder_output)
            
    # Get pooled output if needed
    pooled_output = None
    if add_pooled_output:
        if pooled_text_encoder_output is not None:
            pooled_output = pooled_text_encoder_output
        else:
            # Get pooled output from last hidden state if no direct pooler
            if hasattr(outputs, 'pooler_output'):
                pooled_output = outputs.pooler_output
            else:
                # Use mean pooling on last hidden state as fallback
                pooled_output = outputs.last_hidden_state.mean(dim=1)
            
    return text_encoder_output, pooled_output
