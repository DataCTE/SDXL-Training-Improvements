"""DDPM trainer implementation for SDXL with extreme speedups."""
import logging
import torch
import torch.backends.cudnn
from src.training.methods.base import make_picklable
from src.core.history import TorchHistory
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor


from src.core.memory import torch_sync, create_stream_context
from src.training.methods.base import TrainingMethod
from src.training.schedulers import get_add_time_ids

logger = logging.getLogger(__name__)

class DDPMTrainer(TrainingMethod):
    name = "ddpm"

    def __init__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')

        super().__init__(*args, **kwargs)
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
        if hasattr(torch, "compile"):
            self._compiled_loss = torch.compile(
                self._compute_loss_impl,
                mode="reduce-overhead",
                fullgraph=False
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the history attribute from pickling
        state.pop('history', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the history instance and re-add the hook
        self.history = TorchHistory(self.unet)
        self.history.add_log_parameters_hook()
    @make_picklable
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(model, batch, generator)
        return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        try:
            logger.info("\n=== Starting DDPM Loss Computation ===")
            
            # First validate model input with detailed error
            if "model_input" not in batch:
                error_msg = "Missing model_input in batch"
                logger.error(error_msg, exc_info=True, stack_info=True, extra={
                    'batch_keys': list(batch.keys()),
                    'error_type': 'KeyError',
                    'function': '_compute_loss_impl'
                })
                raise KeyError(error_msg)

            # Log initial batch contents with structured logging
            batch_info = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_info[key] = {
                        'type': 'tensor',
                        'shape': tuple(value.shape),
                        'dtype': str(value.dtype),
                        'device': str(value.device)
                    }
                elif isinstance(value, dict):
                    batch_info[key] = {
                        'type': 'dict',
                        'contents': {
                            k: {
                                'type': 'tensor',
                                'shape': tuple(v.shape),
                                'dtype': str(v.dtype),
                                'device': str(v.device)
                            } if isinstance(v, torch.Tensor) else {'type': str(type(v))}
                            for k, v in value.items()
                        }
                    }
                elif isinstance(value, list):
                    batch_info[key] = {
                        'type': 'list',
                        'length': len(value),
                        'first_element_type': str(type(value[0])) if value else None
                    }

            logger.info("Batch structure:", extra={'batch_info': batch_info})

            # Process latents with validation
            try:
                if isinstance(batch["model_input"], list):
                    latents_list = batch["model_input"]
                else:
                    latents_list = [lat for lat in batch["model_input"]]

                if not latents_list:
                    raise ValueError("Empty latents list")

                for idx, lat in enumerate(latents_list):
                    if not isinstance(lat, torch.Tensor):
                        raise TypeError(f"Latent {idx} is not a tensor: {type(lat)}")
                    if lat.dim() != 4:
                        raise ValueError(f"Latent {idx} has wrong dimensions: {lat.shape}")

            except Exception as e:
                logger.error("Latent processing failed", exc_info=True, stack_info=True, extra={
                    'error_type': type(e).__name__,
                    'latents_info': {
                        'type': type(batch["model_input"]).__name__,
                        'content_type': type(latents_list[0]).__name__ if latents_list else None
                    }
                })
                raise

            # Process embeddings with detailed validation
            try:
                prompt_embeds = None
                pooled_prompt_embeds = None
                
                # Try direct format
                if "prompt_embeds" in batch and "pooled_prompt_embeds" in batch:
                    prompt_embeds = batch["prompt_embeds"]
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                    embedding_source = "direct"
                # Try nested format
                elif "embeddings" in batch:
                    embeddings = batch["embeddings"]
                    prompt_embeds = embeddings.get("prompt_embeds")
                    pooled_prompt_embeds = embeddings.get("pooled_prompt_embeds")
                    embedding_source = "nested"
                else:
                    embedding_source = "missing"

                if prompt_embeds is None or pooled_prompt_embeds is None:
                    error_context = {
                        'embedding_source': embedding_source,
                        'batch_keys': list(batch.keys()),
                        'embeddings_present': 'embeddings' in batch,
                        'direct_embeds_present': all(k in batch for k in ['prompt_embeds', 'pooled_prompt_embeds']),
                        'batch_structure': batch_info
                    }
                    logger.error("Missing required embeddings", exc_info=True, stack_info=True, extra=error_context)
                    raise KeyError("Missing required embeddings - see logs for details")

                # Validate tensor properties
                for name, tensor in [("prompt_embeds", prompt_embeds), 
                                   ("pooled_prompt_embeds", pooled_prompt_embeds)]:
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"{name} must be a tensor, got {type(tensor)}")

            except Exception as e:
                logger.error("Embedding processing failed", exc_info=True, stack_info=True, extra={
                    'error_type': type(e).__name__,
                    'error_msg': str(e),
                    'embedding_source': embedding_source,
                    'batch_structure': batch_info
                })
                raise

            # Process sizes
            logger.info("\nProcessing Size Information:")
            original_sizes = batch.get("original_sizes", [(1024, 1024)] * len(latents_list))
            crop_top_lefts = batch.get("crop_top_lefts", [(0, 0)] * len(latents_list))
            target_sizes = batch.get("target_sizes", [(1024, 1024)] * len(latents_list))

            logger.info(f"Original sizes: {original_sizes}")
            logger.info(f"Crop top lefts: {crop_top_lefts}")
            logger.info(f"Target sizes: {target_sizes}")

            # Stack latents
            logger.info("\nStacking Latents:")
            latents = torch.stack(latents_list, dim=0) if isinstance(batch["model_input"], list) else batch["model_input"]
            logger.info(f"Stacked latents shape: {latents.shape}, dtype={latents.dtype}, device={latents.device}")

            # Generate noise
            logger.info("\nGenerating Noise:")
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator
            )
            logger.info(f"Generated noise shape: {noise.shape}, dtype={noise.dtype}, device={noise.device}")

            # Get timesteps
            logger.info("\nGenerating Timesteps:")
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            logger.info(f"Timesteps shape: {timesteps.shape}, dtype={timesteps.dtype}, device={timesteps.device}")

            # Add noise to latents
            logger.info("\nAdding Noise to Latents:")
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            logger.info(f"Noisy latents shape: {noisy_latents.shape}, dtype={noisy_latents.dtype}, device={noisy_latents.device}")

            # Get time embeddings
            logger.info("\nGenerating Time Embeddings:")
            add_time_ids = get_add_time_ids(
                original_sizes,
                crop_top_lefts, 
                target_sizes,
                dtype=prompt_embeds.dtype,
                device=latents.device
            )
            logger.info(f"Time embeddings shape: {add_time_ids.shape}, dtype={add_time_ids.dtype}, device={add_time_ids.device}")

            # Get model prediction
            logger.info("\nGetting Model Prediction:")
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }
            ).sample
            logger.info(f"Model prediction shape: {noise_pred.shape}, dtype={noise_pred.dtype}, device={noise_pred.device}")

            # Compute loss
            logger.info("\nComputing Loss:")
            if self.config.training.prediction_type == "epsilon":
                logger.info("Using epsilon prediction type")
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                logger.info("Using v_prediction type")
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                logger.error(f"Unknown prediction type {self.config.training.prediction_type}", exc_info=True, stack_info=True)
                raise ValueError(f"Unknown prediction type {self.config.training.prediction_type}")

            logger.info(f"Target shape: {target.shape}, dtype={target.dtype}, device={target.device}")
            
            loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3])
            logger.info(f"Initial loss shape: {loss.shape}, dtype={loss.dtype}, device={loss.device}")

            if "loss_weights" in batch:
                logger.info("Applying loss weights")
                loss = loss * batch["loss_weights"]
                logger.info(f"Weighted loss shape: {loss.shape}")

            loss = loss.mean()
            logger.info(f"Final loss value: {loss.item()}")

            torch_sync()
            return {"loss": loss}

        except Exception as e:
            logger.error(f"DDPM loss computation failed: {str(e)}", exc_info=True, stack_info=True, extra={
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'batch_structure': batch_info if 'batch_info' in locals() else None,
                'traceback': traceback.format_exc()
            })
            raise
