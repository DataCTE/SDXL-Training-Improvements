"""DDPM trainer implementation for SDXL with extreme speedups."""
import logging
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union
from torch import Tensor

from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add verification print
print(f"DDPM Trainer logger initialized with level: {logger.getEffectiveLevel()}")
print(f"DDPM Trainer logger handlers: {logger.handlers}")

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing DDPMTrainer")
        if hasattr(torch, "compile"):
            logger.debug("Compiling loss computation function")
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(batch, generator)
        return self._compute_loss_impl(batch, generator)

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss with detailed shape logging."""
        try:
            # Validate batch tensor shapes
            def validate_tensor_shapes(data: Union[torch.Tensor, Dict], path: str = "") -> None:
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_path = f"{path}.{key}" if path else key
                        validate_tensor_shapes(value, current_path)
                elif isinstance(data, torch.Tensor):
                    logger.debug(f"Tensor at {path}: shape={data.shape}, dtype={data.dtype}, device={data.device}")
                    if path.endswith("model_input") or "latent" in path:
                        # Check if this is the first latent shape we've seen
                        if not hasattr(self, '_expected_latent_shape'):
                            self._expected_latent_shape = data.shape
                        else:
                            if data.shape != self._expected_latent_shape:
                                raise ValueError(
                                    f"Inconsistent latent shapes in batch: "
                                    f"expected {self._expected_latent_shape}, got {data.shape} at {path}"
                                )

            logger.debug("=== Validating Batch Tensor Shapes ===")
            validate_tensor_shapes(batch)

            # Log initial batch structure
            logger.debug("=== Initial Batch Structure ===")
            for key, value in batch.items():
                if isinstance(value, dict):
                    logger.debug(f"Dict {key} contains:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            logger.debug(f"  → {key}.{sub_key}: {sub_value.shape}")
                elif isinstance(value, torch.Tensor):
                    logger.debug(f"→ {key}: {value.shape}")

            # Extract latents with detailed logging
            logger.debug("=== Extracting Latents ===")
            if "latent" in batch:
                logger.debug(f"Latent dict keys: {batch['latent'].keys()}")
                if "model_input" in batch["latent"]:
                    latents = batch["latent"]["model_input"]
                    logger.debug(f"→ Direct model_input shape: {latents.shape}")
                elif "latent" in batch["latent"] and "model_input" in batch["latent"]["latent"]:
                    latents = batch["latent"]["latent"]["model_input"]
                    logger.debug(f"→ Nested model_input shape: {latents.shape}")
                else:
                    logger.debug("Available keys in latent:")
                    for k, v in batch["latent"].items():
                        if isinstance(v, dict):
                            logger.debug(f"  {k}: {v.keys()}")
                        else:
                            logger.debug(f"  {k}: {type(v)}")
                    raise ValueError("Could not find model_input in latent data")
            else:
                latents = batch.get("model_input")
                if latents is None:
                    raise KeyError("No latent data found in batch")
                logger.debug(f"→ Fallback model_input shape: {latents.shape}")

            # Process batch items with shape validation
            logger.debug("=== Processing Batch Items ===")
            if isinstance(latents, list):
                logger.debug(f"Latents is a list of length {len(latents)}")
                for i, lat in enumerate(latents):
                    logger.debug(f"→ Latent[{i}] shape: {lat.shape}")
            elif isinstance(latents, torch.Tensor):
                logger.debug(f"→ Latents tensor shape: {latents.shape}")
            else:
                logger.debug(f"→ Unexpected latents type: {type(latents)}")

            # Generate noise with shape validation
            logger.debug("=== Generating Noise ===")
            try:
                noise = torch.randn_like(latents, generator=generator)
                logger.debug(f"→ Generated noise shape: {noise.shape}")
            except Exception as e:
                logger.error(f"Error generating noise: {str(e)}")
                raise

            # Generate timesteps
            logger.debug("=== Generating Timesteps ===")
            try:
                batch_size = latents.shape[0] if isinstance(latents, torch.Tensor) else len(latents)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device if isinstance(latents, torch.Tensor) else latents[0].device
                )
                logger.debug(f"→ Timesteps shape: {timesteps.shape}")
            except Exception as e:
                logger.error(f"Error generating timesteps: {str(e)}")
                raise

            # Add noise to latents
            logger.debug("=== Adding Noise to Latents ===")
            try:
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                logger.debug(f"→ Noisy latents shape: {noisy_latents.shape}")
            except Exception as e:
                logger.error(f"Error adding noise: {str(e)}")
                raise

            # Extract embeddings with shape validation
            logger.debug("=== Extracting Embeddings ===")
            if "embeddings" in batch:
                logger.debug(f"Embeddings dict keys: {batch['embeddings'].keys()}")
                prompt_embeds = batch["embeddings"].get("prompt_embeds")
                pooled_prompt_embeds = batch["embeddings"].get("pooled_prompt_embeds")
                if prompt_embeds is not None:
                    logger.debug(f"→ Prompt embeds shape: {prompt_embeds.shape}")
                if pooled_prompt_embeds is not None:
                    logger.debug(f"→ Pooled prompt embeds shape: {pooled_prompt_embeds.shape}")
            else:
                prompt_embeds = batch.get("prompt_embeds")
                pooled_prompt_embeds = batch.get("pooled_prompt_embeds")
                logger.debug(f"→ Direct prompt embeds shape: {getattr(prompt_embeds, 'shape', None)}")
                logger.debug(f"→ Direct pooled prompt embeds shape: {getattr(pooled_prompt_embeds, 'shape', None)}")

            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Missing required embeddings")

            # Get add_time_ids with detailed logging
            logger.debug("=== Processing Time IDs ===")
            logger.debug(f"Original sizes: {batch.get('original_sizes')}")
            logger.debug(f"Crop top lefts: {batch.get('crop_top_lefts')}")
            logger.debug(f"Target sizes: {batch.get('target_sizes')}")
            
            try:
                add_time_ids = get_add_time_ids(
                    original_sizes=batch.get("original_sizes", [(1024, 1024)]),
                    crop_top_lefts=batch.get("crop_top_lefts", [(0, 0)]),
                    target_sizes=batch.get("target_sizes", [(1024, 1024)]),
                    dtype=prompt_embeds.dtype,
                    device=prompt_embeds.device
                )
                logger.debug(f"→ Add time ids shape: {add_time_ids.shape}")
            except Exception as e:
                logger.error(f"Error generating time IDs: {str(e)}")
                raise

            # UNet forward pass
            logger.debug("=== UNet Forward Pass ===")
            try:
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "time_ids": add_time_ids,
                        "text_embeds": pooled_prompt_embeds
                    },
                ).sample
                logger.debug(f"→ Predicted noise shape: {noise_pred.shape}")
            except Exception as e:
                logger.error(f"Error in UNet forward pass: {str(e)}")
                raise

            # Compute loss
            logger.debug("=== Computing Loss ===")
            try:
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                logger.debug(f"→ Final loss value: {loss.item():.6f}")
            except Exception as e:
                logger.error(f"Error computing loss: {str(e)}")
                raise

            return {"loss": loss}

        except Exception as e:
            logger.error("=== Error State Tensor Shapes ===")
            # Log all tensor shapes in the batch
            logger.error("Batch tensor shapes:")
            for key, value in batch.items():
                if isinstance(value, dict):
                    logger.error(f"{key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            logger.error(f"  → {key}.{sub_key}: {sub_value.shape}")
                elif isinstance(value, torch.Tensor):
                    logger.error(f"→ {key}: {value.shape}")

            # Log local variables' shapes
            logger.error("\nLocal variable tensor shapes:")
            local_vars = locals()
            for name, var in local_vars.items():
                if isinstance(var, torch.Tensor):
                    logger.error(f"→ {name}: {var.shape}")
                elif isinstance(var, list) and var and isinstance(var[0], torch.Tensor):
                    logger.error(f"→ {name}: [list of tensors with shapes: {[t.shape for t in var]}]")

            # Log full traceback
            logger.error("\nFull traceback:", exc_info=True)
            
            # Log additional context
            logger.error("\nError context:", extra={
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'stack_info': True
            })
            
            raise RuntimeError(f"Loss computation failed with detailed shapes logged above: {str(e)}") from e
