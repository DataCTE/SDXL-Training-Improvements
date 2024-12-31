"""Flow Matching trainer implementation with extreme speedups."""
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from torch import Tensor
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import time

from src.core.logging import get_logger, MetricsLogger
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.training.schedulers import get_add_time_ids
from src.data.config import Config
from src.core.distributed import is_main_process
from src.core.logging import WandbLogger
from src.core.types import DataType, ModelWeightDtypes
from src.models.sdxl import StableDiffusionXL

logger = get_logger(__name__)

class FlowMatchingTrainer(SDXLTrainer):
    """Flow Matching trainer with memory optimizations."""
    
    name = "flow_matching"

    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        wandb_logger: Optional[WandbLogger] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            device=device,
            wandb_logger=wandb_logger,
            config=config,
            **kwargs
        )
        
        # Verify effective batch size with gradient accumulation
        self.effective_batch_size = (
            config.training.batch_size * 
            self.gradient_accumulation_steps
        )
        logger.info(
            f"Effective batch size: {self.effective_batch_size} "
            f"(batch_size={config.training.batch_size} × "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps})"
        )
        
        # Enable memory optimizations on individual components
        if hasattr(self.model.unet, "enable_gradient_checkpointing"):
            logger.info("Enabling gradient checkpointing for UNet")
            self.model.unet.enable_gradient_checkpointing()
        
        if hasattr(self.model.vae, "enable_gradient_checkpointing"):
            logger.info("Enabling gradient checkpointing for VAE")
            self.model.vae.enable_gradient_checkpointing()
        
        # Enable xformers memory efficient attention if available
        if config.training.enable_xformers:
            logger.info("Enabling xformers memory efficient attention")
            if hasattr(self.model.unet, "enable_xformers_memory_efficient_attention"):
                self.model.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers for UNet")
            if hasattr(self.model.vae, "enable_xformers_memory_efficient_attention"):
                self.model.vae.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers for VAE")
        
        # Move model to device
        self.model.to(device)
        
        # Enable CPU offload if available
        if hasattr(self.model, "enable_model_cpu_offload"):
            logger.info("Enabling model CPU offload")
            self.model.enable_model_cpu_offload()
        
        # Set up mixed precision training
        self.mixed_precision = config.training.mixed_precision
        if self.mixed_precision != "no":
            self.scaler = GradScaler()
        
        # Handle optimizer-specific dtype conversions
        if config.optimizer.optimizer_type == "adamw_bf16":
            logger.info("Converting model to bfloat16 format for AdamWBF16 optimizer")
            bfloat16_weights = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
            self.model.unet.to(bfloat16_weights.unet.to_torch_dtype())
            self.model.vae.to(bfloat16_weights.vae.to_torch_dtype())
            self.model.text_encoder_1.to(bfloat16_weights.text_encoder.to_torch_dtype())
            self.model.text_encoder_2.to(bfloat16_weights.text_encoder_2.to_torch_dtype())
            model_dtype = torch.bfloat16
        else:
            model_dtype = next(self.model.parameters()).dtype
        
        logger.info(f"Model components using dtype: {model_dtype}")

        # Try to compile loss computation if available
        if hasattr(torch, "compile"):
            self.logger.debug("Attempting to compile loss computation")
            try:
                self._compiled_loss = torch.compile(
                    self._compute_loss_impl,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                self.logger.debug("Loss computation successfully compiled")
            except Exception as e:
                self.logger.warning(
                    "Failed to compile loss computation",
                    exc_info=True,
                    extra={'error': str(e)}
                )

    def train(self, num_epochs: int):
        """Execute training loop for specified number of epochs."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting Flow Matching training with {total_steps} total steps ({num_epochs} epochs)")
        
        # Initialize progress tracking
        global_step = 0
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc="Training Flow Matching",
            position=0,
            leave=True
        )
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                self.model.train()
                # Track accumulated loss
                accumulated_loss = 0.0
                accumulated_metrics = defaultdict(float)
                
                for step, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    # Training step
                    loss, metrics = self._execute_training_step(
                        batch,
                        accumulate=True,
                        is_last_accumulation_step=((step + 1) % self.config.training.gradient_accumulation_steps == 0)
                    )
                    
                    step_time = time.time() - step_start_time
                    
                    # Update progress bar
                    progress_bar.set_postfix(
                        {'Loss': f"{loss.item():.4f}", 'Time': f"{step_time:.1f}s"},
                        refresh=True
                    )
                    
                    # Accumulate loss and metrics
                    accumulated_loss += loss.item()
                    for k, v in metrics.items():
                        accumulated_metrics[k] += v
                        
                    # Only update weights after accumulating enough gradients
                    if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                        # Scale loss and metrics by accumulation steps
                        effective_loss = accumulated_loss / self.config.training.gradient_accumulation_steps
                        effective_metrics = {
                            k: v / self.config.training.gradient_accumulation_steps 
                            for k, v in accumulated_metrics.items()
                        }
                        
                        # Update weights
                        if self.config.training.clip_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.training.clip_grad_norm
                            )
                            effective_metrics['grad_norm'] = grad_norm.item()
                        
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Reset accumulators
                        accumulated_loss = 0.0
                        accumulated_metrics.clear()
                        
                        # Log metrics
                        if is_main_process() and self.wandb_logger:
                            self.wandb_logger.log_metrics(
                                {
                                    'epoch': epoch + 1,
                                    'step': global_step,
                                    'loss': effective_loss,
                                    'step_time': step_time,
                                    **effective_metrics
                                },
                                step=global_step
                            )
                    
                    global_step += 1
                    progress_bar.update(1)
                
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch + 1}, step {step + 1}", exc_info=True)
            raise
        finally:
            progress_bar.close()

    def _execute_training_step(self, batch, accumulate: bool = False, is_last_accumulation_step: bool = True):
        """Execute single training step with gradient accumulation support."""
        try:
            # Don't zero gradients when accumulating
            if not accumulate or is_last_accumulation_step:
                self.optimizer.zero_grad()
            
            step_output = self.compute_loss(self.model, batch)
            loss = step_output["loss"]
            metrics = step_output["metrics"]
            
            # Scale loss by accumulation steps when accumulating
            if accumulate:
                loss = loss / self.config.training.gradient_accumulation_steps
            
            loss.backward()
            
            # Only return the unscaled loss for logging
            return loss * self.config.training.gradient_accumulation_steps, metrics
            
        except Exception as e:
            logger.error("Flow Matching training step failed", exc_info=True)
            raise

    def compute_loss(self, model, batch, generator=None) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(model, batch, generator)
        return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute Flow Matching loss with new data format."""
        try:
            # Get model dtype from parameters
            model_dtype = next(self.model.parameters()).dtype

            # Get batch data and ensure consistent dtype
            pixel_values = batch["pixel_values"].to(self.device, dtype=model_dtype)
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=model_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=model_dtype)
            original_sizes = batch["original_sizes"]
            crop_top_lefts = batch["crop_top_lefts"]

            # Convert images to latent space using VAE
            latents = self.model.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor

            # Use latents as target (x1)
            x1 = latents

            # Sample time steps
            t = self.sample_logit_normal(
                (x1.shape[0],),
                x1.device,
                x1.dtype,
                generator=generator
            )

            # Generate random starting point with matching dtype
            x0 = torch.randn_like(x1, device=self.device, dtype=model_dtype)

            # Prepare time embeddings for SDXL
            time_ids = self._get_add_time_ids(
                original_sizes=original_sizes,
                crops_coords_top_left=crop_top_lefts,
                target_size=(pixel_values.shape[-2], pixel_values.shape[-1]),
                dtype=prompt_embeds.dtype,
                batch_size=x1.shape[0]
            )
            time_ids = time_ids.to(device=self.device)

            # Prepare conditioning embeddings
            cond_emb = {
                "prompt_embeds": prompt_embeds,
                "added_cond_kwargs": {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids
                }
            }

            # Compute flow matching loss
            loss = self.compute_flow_matching_loss(self.model.unet, x0, x1, t, cond_emb)
            loss = loss.mean()

            # Compute additional metrics
            metrics = {
                "loss": loss.detach().item(),
                "x0_norm": x0.norm().item(),
                "x1_norm": x1.norm().item(),
                "time_mean": t.mean().item(),
                "time_std": t.std().item()
            }

            return {
                "loss": loss,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(
                "Error computing Flow Matching loss",
                exc_info=True,
                extra={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                    'device_info': {
                        'x1_device': x1.device if 'x1' in locals() else None,
                        'prompt_device': prompt_embeds.device if 'prompt_embeds' in locals() else None
                    }
                }
            )
            raise

    def _log_tensor_shapes(self, tensors: Dict[str, Tensor], step: str):
        """Log tensor shapes for debugging."""
        shapes = {
            name: tensor.shape if hasattr(tensor, 'shape') else None 
            for name, tensor in tensors.items()
        }
        self._shape_logs.append({
            'step': step,
            'shapes': shapes,
            'devices': {
                name: str(tensor.device) if hasattr(tensor, 'device') else None
                for name, tensor in tensors.items()
            }
        })

    def sample_logit_normal(
        self, 
        shape, 
        device, 
        dtype, 
        mean=0.0, 
        std=1.0, 
        generator=None
    ) -> Tensor:
        """Sample from logit-normal distribution."""
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        return torch.sigmoid(normal)

    def optimal_transport_path(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Compute optimal transport path."""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_velocity(
        self, 
        model: torch.nn.Module, 
        xt: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute velocity field."""
        return model(
            xt,
            t,
            encoder_hidden_states=cond_emb["prompt_embeds"],
            added_cond_kwargs=cond_emb["added_cond_kwargs"]
        ).sample

    def compute_flow_matching_loss(
        self, 
        model: torch.nn.Module, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute flow matching loss."""
        xt = self.optimal_transport_path(x0, x1, t)
        v_true = x1 - x0
        v_pred = self.compute_velocity(model, xt, t, cond_emb)
        return F.mse_loss(v_pred, v_true, reduction="none").mean([1, 2, 3])
