"""CLIP encoder implementation with extreme speedups and embedding support."""
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import random
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from src.core.logging import get_logger

logger = get_logger(__name__)

class CLIPEncoder:
    """Optimized CLIP encoder wrapper."""
    
    def __init__(self, text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection]):
        """Initialize CLIP encoder.
        
        Args:
            text_encoder: CLIP text encoder model (either CLIPTextModel or CLIPTextModelWithProjection)
        """
        if not isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            raise TypeError(f"text_encoder must be CLIPTextModel or CLIPTextModelWithProjection, got {type(text_encoder)}")
        self.text_encoder = text_encoder

    @staticmethod
    @torch.inference_mode()
    def encode_prompt(
        batch: Dict[str, Any],
        text_encoders: List[torch.nn.Module],
        tokenizers: List[CLIPTokenizer],
        proportion_empty_prompts: float = 0.0,
        caption_column: str = "text",
        is_train: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts using both CLIP encoders.
        
        Args:
            batch: Dictionary containing prompts under caption_column
            text_encoders: List of CLIP text encoders
            tokenizers: List of CLIP tokenizers
            proportion_empty_prompts: Proportion of prompts to replace with empty strings
            caption_column: Column name containing prompts
            is_train: Whether in training mode
            
        Returns:
            Dict containing prompt_embeds and pooled_prompt_embeds
        """
        prompt_embeds_list = []
        prompt_batch = batch[caption_column]

        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
                return_dict=False,
            )

            # Keep everything on GPU
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        # Concatenate on GPU
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        
        # Only move to CPU if explicitly needed
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds
        }

