"""CLIP encoder implementation with extreme speedups."""
from typing import Dict, List, Optional, Tuple, Union
import logging
import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer
from src.core.logging.logging import setup_logging

# Initialize logger with debug disabled by default
logger = setup_logging(__name__, level=logging.INFO)

class CLIPEncoder:
    """Optimized CLIP encoder wrapper with extreme speedup."""
    
    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16,
        debug: bool = False
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.debug = debug
        
        # Apply optimizations
        self.text_encoder.to(device=self.device, dtype=self.dtype)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            self.text_encoder = torch.compile(self.text_encoder, mode="reduce-overhead", fullgraph=False)
            
        if self.debug:
            logger.setLevel("DEBUG")
            logger.debug("CLIP encoder debug logging enabled")
            
        logger.info("CLIP encoder initialized", extra={
            'device': str(self.device),
            'dtype': str(self.dtype),
            'model_type': type(self.text_encoder).__name__
        })

    def encode_prompt(self, prompt_batch: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts using CLIP encoder.
        
        Args:
            prompt_batch: List of text prompts to encode
            
        Returns:
            Dictionary containing text embeddings and pooled outputs
        """
        try:
            if self.debug:
                logger.debug("Starting prompt encoding", extra={
                    'batch_size': len(prompt_batch),
                    'device': str(self.device)
                })
            
            # Tokenize prompts
            tokens = self.tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=self.text_encoder.config.max_position_embeddings,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move tokens to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Encode tokens
            text_encoder_output, pooled_output = self.encode(
                tokens=tokens["input_ids"],
                add_pooled_output=True
            )
            
            result = {
                "text_embeds": text_encoder_output,
                "pooled_embeds": pooled_output
            }
            
            if self.debug:
                logger.debug("Prompt encoding complete", extra={
                    'output_shapes': {
                        'text_embeds': tuple(text_encoder_output.shape),
                        'pooled_embeds': tuple(pooled_output.shape) if pooled_output is not None else None
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error("Failed to encode prompts", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'batch_size': len(prompt_batch),
                'stack_trace': True
            })
            raise

    def encode(
        self,
        tokens: Tensor,
        default_layer: int = -2,
        layer_skip: int = 0,
        text_encoder_output: Optional[Tensor] = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Optional[Tensor] = None,
        use_attention_mask: bool = False,
        add_layer_norm: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Low-level CLIP encoding with caching support."""
        try:
            if self.debug:
                logger.debug("Starting CLIP encoding", extra={
                    'tokens_shape': tuple(tokens.shape),
                    'default_layer': default_layer,
                    'layer_skip': layer_skip,
                    'use_attention_mask': use_attention_mask
                })

            # Ensure tokens are on correct device and dtype
            tokens = tokens.to(self.device, dtype=torch.long)

            if text_encoder_output is None:
                attention_mask = None
                if use_attention_mask:
                    attention_mask = tokens.ne(self.text_encoder.config.pad_token_id).long()

                with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
                    outputs = self.text_encoder(
                        tokens,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    target_layer = default_layer - layer_skip
                    text_encoder_output = outputs.hidden_states[target_layer]

                    if add_layer_norm and hasattr(self.text_encoder, "final_layer_norm"):
                        text_encoder_output = self.text_encoder.final_layer_norm(text_encoder_output)

                    pooled_output = None
                    if add_pooled_output:
                        if pooled_text_encoder_output is not None:
                            pooled_output = pooled_text_encoder_output
                        else:
                            pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)

            if self.debug:
                logger.debug("CLIP encoding complete", extra={
                    'output_shape': tuple(text_encoder_output.shape),
                    'pooled_shape': tuple(pooled_output.shape) if pooled_output is not None else None
                })

            return text_encoder_output, pooled_output

        except Exception as e:
            logger.error("CLIP encoding failed", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'tokens_shape': tuple(tokens.shape) if isinstance(tokens, Tensor) else None,
                'stack_trace': True
            })
            raise
