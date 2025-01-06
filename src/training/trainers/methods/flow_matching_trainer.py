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

from src.core.logging import UnifiedLogger, LogConfig, MetricsTracker
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.training.schedulers import get_add_time_ids
from src.data.config import Config
from src.core.distributed import is_main_process
from src.core.logging import WandbLogger
from src.core.types import DataType, ModelWeightDtypes
from src.models.sdxl import StableDiffusionXL

logger = UnifiedLogger(LogConfig(name=__name__))

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
        best_loss = float('inf')
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
                epoch_loss = 0.0
                accumulated_loss = 0.0
                accumulated_metrics = defaultdict(float)
                valid_steps = 0
                
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
                    epoch_loss += loss.item()
                    valid_steps += 1
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
                
                # Compute epoch average loss
                if valid_steps > 0:
                    avg_epoch_loss = epoch_loss / valid_steps
                    
                    # Save checkpoint if loss improved
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        if is_main_process():
                            self.save_checkpoint(epoch + 1, is_final=False)
                            logger.info(f"Saved checkpoint for epoch {epoch + 1} with loss {best_loss:.4f}")
                
                # Clear CUDA cache between epochs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch + 1}", exc_info=True)
            raise
        finally:
            progress_bar.close()
            
            # Save final checkpoint
            if is_main_process():
                self.save_checkpoint(num_epochs, is_final=True)
                logger.info("Saved final checkpoint")

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
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute Flow Matching loss with enhanced bucket metadata."""
        try:
            # Validate batch contents
            required_keys = {
                "vae_latents", "prompt_embeds", "pooled_prompt_embeds", 
                "time_ids", "metadata"
            }
            if not all(k in batch for k in required_keys):
                missing = required_keys - set(batch.keys())
                raise ValueError(f"Batch missing required keys: {missing}")

            # Get model dtype from parameters
            model_dtype = next(self.model.parameters()).dtype

            # Extract and validate tensors
            vae_latents = batch["vae_latents"].to(self.device, dtype=model_dtype)
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=model_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=model_dtype)
            time_ids = batch["time_ids"].to(self.device, dtype=model_dtype)
            batch_size = vae_latents.shape[0]

            # Use VAE latents as target (x1)
            x1 = vae_latents

            # Sample time steps from logit-normal distribution
            t = self.sample_logit_normal(
                (batch_size,),
                self.device,
                model_dtype,
                generator=generator
            )

            # Generate random starting point (x0)
            x0 = torch.randn_like(x1, device=self.device, dtype=model_dtype)

            # Prepare conditioning embeddings
            cond_emb = {
                "prompt_embeds": prompt_embeds,
                "added_cond_kwargs": {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids
                }
            }

            # Compute optimal transport path and velocity field
            xt = self.optimal_transport_path(x0, x1, t)
            v = self.compute_velocity(model, xt, t, cond_emb)

            # Compute flow matching loss
            loss = self.compute_flow_matching_loss(model.unet, x0, x1, t, cond_emb)
            loss = loss.mean()

            # Apply tag weights if present
            if "tag_weights" in batch:
                tag_weights = batch["tag_weights"].to(self.device, dtype=model_dtype)
                loss = loss * tag_weights.mean()

            # Ensure loss is finite and properly scaled
            if not torch.isfinite(loss):
                logger.warning("Invalid loss detected - using fallback value")
                loss = torch.tensor(1000.0, device=self.device, dtype=model_dtype)
            else:
                loss = torch.clamp(loss, max=1000.0)

            # Compute additional metrics for monitoring
            metrics = {
                "loss": loss.detach().item(),
                "x0_norm": x0.norm().item(),
                "x1_norm": x1.norm().item(),
                "time_mean": t.mean().item(),
                "time_std": t.std().item(),
                "velocity_norm": v.norm().item(),
                "batch_size": batch_size,
                "lr": self.optimizer.param_groups[0]["lr"]
            }

            return {
                "loss": loss,
                "metrics": metrics
            }

        except Exception as e:
            logger.error("Error computing Flow Matching loss", exc_info=True)
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
    
    def compute_time_ids(self, original_size, crops_coords_top_left, target_size, device=None, dtype=None):
        """Compute time embeddings for SDXL.
        
        Args:
            original_size (tuple): Original image size (height, width)
            crops_coords_top_left (tuple): Crop coordinates (top, left)
            target_size (tuple): Target image size (height, width)
            device (torch.device, optional): Device to place tensor on
            dtype (torch.dtype, optional): Dtype for the tensor
        
        Returns:
            torch.Tensor: Time embeddings tensor of shape [1, 6]
        """
        # Ensure inputs are proper tuples
        if not isinstance(original_size, (tuple, list)):
            original_size = (original_size, original_size)
        if not isinstance(crops_coords_top_left, (tuple, list)):
            crops_coords_top_left = (crops_coords_top_left, crops_coords_top_left)
        if not isinstance(target_size, (tuple, list)):
            target_size = (target_size, target_size)
        
        # Combine all values into a single list
        time_ids = [
            original_size[0],    # Original height
            original_size[1],    # Original width
            crops_coords_top_left[0],  # Crop top
            crops_coords_top_left[1],  # Crop left
            target_size[0],     # Target height
            target_size[1],     # Target width
        ]
        
        # Create tensor with proper device and dtype
        return torch.tensor([time_ids], device=device, dtype=dtype)
