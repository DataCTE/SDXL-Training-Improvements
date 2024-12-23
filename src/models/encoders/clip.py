"""CLIP model encoding utilities with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
from torch import Tensor
from transformers import CLIPTextModel
from typing import Optional, Tuple

# Force maximum speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

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
    add_layer_norm: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    if text_encoder_output is None:
        attention_mask = None
        if use_attention_mask:
            attention_mask = tokens.ne(text_encoder.config.pad_token_id).long()

        with torch.inference_mode():
            outputs = text_encoder(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        target_layer = default_layer - layer_skip
        text_encoder_output = outputs.hidden_states[target_layer]

        if add_layer_norm and hasattr(text_encoder, "final_layer_norm"):
            text_encoder_output = text_encoder.final_layer_norm(text_encoder_output)

    pooled_output = None
    if add_pooled_output:
        if pooled_text_encoder_output is not None:
            pooled_output = pooled_text_encoder_output
        else:
            if hasattr(outputs, "pooler_output"):
                pooled_output = outputs.pooler_output
            else:
                pooled_output = outputs.last_hidden_state.mean(dim=1)

    return text_encoder_output, pooled_output
