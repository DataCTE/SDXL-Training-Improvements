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
            
            # Log initial batch contents
            logger.info("\nInput Batch Contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                elif isinstance(value, dict):
                    logger.info(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                elif isinstance(value, list):
                    logger.info(f"{key}: list of length {len(value)}")
                    if value and isinstance(value[0], torch.Tensor):
                        logger.info(f"  First element: shape={value[0].shape}, dtype={value[0].dtype}, device={value[0].device}")

            # First validate model input
            if "model_input" not in batch:
                logger.error("Missing model_input in batch", exc_info=True, stack_info=True)
                raise KeyError("Missing model_input in batch")

            # Process latents first
            logger.info("\nProcessing Latents:")
            if isinstance(batch["model_input"], list):
                latents_list = batch["model_input"]
                logger.info("Model input is a list of tensors")
            else:
                latents_list = [lat for lat in batch["model_input"]]
                logger.info("Model input converted to list")

            logger.info(f"Number of latents: {len(latents_list)}")
            for idx, lat in enumerate(latents_list):
                logger.info(f"Latent {idx}: shape={lat.shape}, dtype={lat.dtype}, device={lat.device}")

            # Process embeddings - handle both direct and nested formats
            logger.info("\nProcessing Embeddings:")
            prompt_embeds = None
            pooled_prompt_embeds = None
            
            # Try direct format first
            if "prompt_embeds" in batch and "pooled_prompt_embeds" in batch:
                prompt_embeds = batch["prompt_embeds"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                logger.info("Using direct embedding format")
            # Try nested format
            elif "embeddings" in batch:
                embeddings = batch["embeddings"]
                prompt_embeds = embeddings.get("prompt_embeds")
                pooled_prompt_embeds = embeddings.get("pooled_prompt_embeds")
                logger.info("Using nested embedding format")

            if prompt_embeds is None or pooled_prompt_embeds is None:
                logger.error("\nMissing embeddings - Current batch structure:", exc_info=True, stack_info=True)
                for key, value in batch.items():
                    if isinstance(value, dict):
                        logger.error(f"{key}:")
                        for subkey, subval in value.items():
                            if isinstance(subval, torch.Tensor):
                                logger.error(f"  {subkey}: shape={subval.shape}, dtype={subval.dtype}")
                            else:
                                logger.error(f"  {subkey}: type={type(subval)}")
                    elif isinstance(value, torch.Tensor):
                        logger.error(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        logger.error(f"{key}: type={type(value)}")
                raise KeyError("Missing required embeddings - checked both direct and nested formats")

            # Validate tensor properties
            if not isinstance(prompt_embeds, torch.Tensor):
                raise TypeError(f"prompt_embeds must be a tensor, got {type(prompt_embeds)}")
            if not isinstance(pooled_prompt_embeds, torch.Tensor):
                raise TypeError(f"pooled_prompt_embeds must be a tensor, got {type(pooled_prompt_embeds)}")

            logger.info(f"Found embeddings:")
            logger.info(f"prompt_embeds shape: {prompt_embeds.shape}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}")
            logger.info(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}, dtype={pooled_prompt_embeds.dtype}, device={pooled_prompt_embeds.device}")

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
            logger.error(f"Error in DDPM loss computation: {str(e)}", exc_info=True, stack_info=True)
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise
