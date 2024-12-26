"""CLIP encoder implementation with extreme speedups and embedding support."""
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
import torch
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from src.core.logging.logging import setup_logging
from src.models.embeddings import BaseModelEmbedding

# Initialize logger with debug disabled by default
logger = setup_logging(__name__, level=logging.INFO)

class CLIPEncoder:
    """Optimized CLIP encoder wrapper with extreme speedup."""
    
    def __init__(
        self,
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
        tokenizer: CLIPTokenizer,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = torch.float16,
        enable_memory_efficient_attention: bool = True,
        enable_vae_slicing: bool = False,
        enable_model_cpu_offload: bool = False,
        debug: bool = False
    ):
        """Initialize CLIP encoder with optimizations.
        
        Args:
            text_encoder: CLIP text encoder model
            tokenizer: CLIP tokenizer
            device: Target device
            dtype: Model dtype
            enable_memory_efficient_attention: Whether to use memory efficient attention
            enable_vae_slicing: Whether to enable VAE slicing
            enable_model_cpu_offload: Whether to enable CPU offloading
            debug: Enable debug logging
        """
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.debug = debug
        
        # Store configuration
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_model_cpu_offload = enable_model_cpu_offload
        
        # Apply optimizations
        self.text_encoder.to(device=self.device, dtype=self.dtype)
        
        # Enable memory efficient attention if requested
        if enable_memory_efficient_attention and hasattr(self.text_encoder, "set_attention_slice"):
            self.text_encoder.set_attention_slice(True)
            
        # Apply torch compile optimization for CUDA devices
        if hasattr(torch, "compile") and self.device.type == "cuda":
            self.text_encoder = torch.compile(
                self.text_encoder,
                mode="reduce-overhead",
                fullgraph=False
            )
            
        if self.debug:
            logger.setLevel("DEBUG")
            logger.debug("CLIP encoder debug logging enabled")
            
        logger.info("CLIP encoder initialized", extra={
            'device': str(self.device),
            'dtype': str(self.dtype),
            'model_type': type(self.text_encoder).__name__,
            'optimizations': {
                'memory_efficient_attention': enable_memory_efficient_attention,
                'vae_slicing': enable_vae_slicing,
                'model_cpu_offload': enable_model_cpu_offload,
                'torch_compile': hasattr(torch, "compile")
            }
        })

    def encode_prompt(
        self,
        prompt_batch: List[str],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts using CLIP encoder.
        
        Args:
            prompt_batch: List of text prompts to encode
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            negative_prompt: Optional negative prompt(s)
            prompt_embeds: Optional pre-computed prompt embeddings
            negative_prompt_embeds: Optional pre-computed negative prompt embeddings
            clip_skip: Optional number of layers to skip
            
        Returns:
            Dictionary containing text embeddings and pooled outputs
        """
        try:
            if self.debug:
                logger.debug("Starting prompt encoding", extra={
                    'batch_size': len(prompt_batch),
                    'device': str(self.device),
                    'num_images_per_prompt': num_images_per_prompt,
                    'do_classifier_free_guidance': do_classifier_free_guidance
                })

            # Handle pre-computed embeddings
            if prompt_embeds is not None:
                if negative_prompt_embeds is None and do_classifier_free_guidance:
                    raise ValueError("negative_prompt_embeds must be provided if using prompt_embeds with guidance")
                return {
                    "text_embeds": prompt_embeds,
                    "pooled_embeds": negative_prompt_embeds if do_classifier_free_guidance else None
                }

            # Process prompts
            if isinstance(prompt_batch, str):
                prompt_batch = [prompt_batch]
            
            # Handle negative prompts
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt_batch)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt_batch)
                
            # Tokenize prompts
            text_inputs = self.tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize negative prompts if needed
            if do_classifier_free_guidance:
                uncond_tokens = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )

            # Move tokens to device
            text_input_ids = text_inputs.input_ids.to(self.device)
            if do_classifier_free_guidance:
                uncond_input_ids = uncond_tokens.input_ids.to(self.device)
            
            # Encode tokens
            with torch.inference_mode():
                # Get prompt embeddings
                prompt_embeds, pooled_prompt_embeds = self.encode(
                    tokens=text_input_ids,
                    output_hidden_states=True,
                    output_attentions=False,
                    clip_skip=clip_skip
                )
                
                # Get negative prompt embeddings if needed
                if do_classifier_free_guidance:
                    negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode(
                        tokens=uncond_input_ids,
                        output_hidden_states=True,
                        output_attentions=False,
                        clip_skip=clip_skip
                    )

                    # Concatenate embeddings
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

            # Repeat for each requested image
            bs_embed = prompt_embeds.shape[0]
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, -1, prompt_embeds.shape[-1])
            
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)

            result = {
                "text_embeds": prompt_embeds,
                "pooled_embeds": pooled_prompt_embeds
            }
            
            if self.debug:
                logger.debug("Prompt encoding complete", extra={
                    'output_shapes': {
                        'text_embeds': tuple(prompt_embeds.shape),
                        'pooled_embeds': tuple(pooled_prompt_embeds.shape)
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
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        clip_skip: Optional[int] = None,
        return_dict: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Low-level CLIP encoding with enhanced features.
        
        Args:
            tokens: Input token ids
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
            clip_skip: Number of layers to skip
            return_dict: Whether to return dict or tuple
            
        Returns:
            Tuple of (text embeddings, pooled embeddings)
        """
        try:
            if self.debug:
                logger.debug("Starting CLIP encoding", extra={
                    'tokens_shape': tuple(tokens.shape),
                    'output_hidden_states': output_hidden_states,
                    'clip_skip': clip_skip
                })

            # Ensure tokens are on correct device and dtype
            tokens = tokens.to(self.device, dtype=torch.long)

            with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
                outputs = self.text_encoder(
                    tokens,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=return_dict
                )

                if clip_skip is not None:
                    # Handle CLIP layer skipping
                    if clip_skip > len(outputs.hidden_states):
                        raise ValueError(f"clip_skip ({clip_skip}) must be less than number of layers")
                    text_embeds = outputs.hidden_states[-clip_skip]
                else:
                    # Use default hidden state
                    text_embeds = outputs.hidden_states[-2] if output_hidden_states else outputs.last_hidden_state

                # Get pooled output
                pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else None
                if pooled_output is None and output_hidden_states:
                    # Compute pooled output if not available
                    pooled_output = outputs.last_hidden_state.mean(dim=1)

                # Validate outputs
                for tensor, name in [(text_embeds, "text_embeds"), (pooled_output, "pooled_output")]:
                    if tensor is not None:
                        if torch.isnan(tensor).any():
                            raise ValueError(f"NaN values detected in {name}")
                        if torch.isinf(tensor).any():
                            tensor = torch.clamp(tensor, -1e6, 1e6)

                if self.debug:
                    logger.debug("CLIP encoding complete", extra={
                        'output_shape': tuple(text_embeds.shape),
                        'pooled_shape': tuple(pooled_output.shape) if pooled_output is not None else None
                    })

                return text_embeds, pooled_output

        except Exception as e:
            logger.error("CLIP encoding failed", extra={
                'error_type': type(e).__name__,
                'error': str(e),
                'tokens_shape': tuple(tokens.shape) if isinstance(tokens, Tensor) else None,
                'stack_trace': True
            })
            raise
    def __call__(self, *args, **kwargs):
        """Convenience wrapper around encode_prompt."""
        return self.encode_prompt(*args, **kwargs)
