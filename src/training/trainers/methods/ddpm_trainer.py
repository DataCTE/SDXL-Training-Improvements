"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any, Optional
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import time

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config

logger = get_logger(__name__)

class DDPMTrainer(SDXLTrainer):
    """DDPM-specific trainer implementation."""
    
    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        wandb_logger=None,
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
            config.training.gradient_accumulation_steps
        )
        logger.info(
            f"Effective batch size: {self.effective_batch_size} "
            f"(batch_size={config.training.batch_size} × "
            f"gradient_accumulation_steps={config.training.gradient_accumulation_steps})"
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
        
        self.noise_scheduler = model.noise_scheduler
        
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
        
    def train(self, num_epochs: int):
        """Execute training loop with proper gradient accumulation."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting training with {total_steps} total steps ({num_epochs} epochs)")
        
        # Initialize progress tracking
        global_step = 0
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc=f"Training DDPM ({self.config.training.prediction_type})",
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
                    
                    # Execute training step and accumulate gradients
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
            
            step_output = self.training_step(batch)
            loss = step_output["loss"]
            metrics = step_output["metrics"]
            
            # Scale loss by accumulation steps when accumulating
            if accumulate:
                loss = loss / (self.config.training.gradient_accumulation_steps * self.current_accumulation_factor)
            
            loss.backward()
            
            # Only return the unscaled loss for logging
            return loss * (self.config.training.gradient_accumulation_steps * self.current_accumulation_factor), metrics
            
        except Exception as e:
            logger.error("Training step failed", exc_info=True)
            raise

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single training step with memory optimizations."""
        try:
            # Verify batch size
            actual_batch_size = batch["pixel_values"].shape[0]
            if actual_batch_size != self.config.training.batch_size and is_main_process():
                logger.warning(
                    f"Received batch size {actual_batch_size} differs from "
                    f"configured batch size {self.config.training.batch_size}"
                )
                # Adjust gradient accumulation steps for this batch
                self.current_accumulation_factor = self.config.training.batch_size / actual_batch_size
            else:
                self.current_accumulation_factor = 1.0

            # Clear cache before forward pass
            torch.cuda.empty_cache()
            
            # Get model dtype from parameters
            model_dtype = next(self.model.parameters()).dtype
            
            # Move batch data to device efficiently
            pixel_values = batch["pixel_values"].to(self.device, dtype=model_dtype, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=model_dtype, non_blocking=True)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=model_dtype, non_blocking=True)
            
            # Store image dimensions before processing
            image_height, image_width = pixel_values.shape[-2:]
            
            # Use context manager for mixed precision
            with autocast(device_type='cuda', enabled=self.mixed_precision != "no"):
                # Convert images to latent space
                with torch.no_grad():
                    latents = self.model.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.model.vae.config.scaling_factor
                    
                # Free up memory
                del pixel_values
                torch.cuda.empty_cache()
                
                # Prepare time embeddings for SDXL
                batch_size = latents.shape[0]
                time_ids = self._get_add_time_ids(
                    original_sizes=batch["original_sizes"],
                    crops_coords_top_left=batch["crop_top_lefts"],
                    target_size=(image_height, image_width),  # Use stored dimensions
                    dtype=prompt_embeds.dtype,
                    batch_size=batch_size
                )
                time_ids = time_ids.to(device=self.device)

                # Sample noise and timesteps
                noise = torch.randn_like(latents, device=self.device, dtype=model_dtype)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), device=self.device
                ).long()

                # Add noise to latents
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get model prediction with time embeddings
                model_pred = self.model.unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids
                    }
                ).sample

                # Calculate loss with consistent dtype
                if self.config.training.prediction_type == "epsilon":
                    target = noise
                elif self.config.training.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {self.config.training.prediction_type}")

                # Ensure both tensors are in the same dtype for loss calculation
                loss = F.mse_loss(
                    model_pred.to(dtype=model_dtype),
                    target.to(dtype=model_dtype),
                    reduction="mean"
                )

                # Log metrics
                metrics = {
                    "loss": loss.detach().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "timestep_mean": timesteps.float().mean().item(),
                }
                
                # Only calculate std if batch size > 1 to avoid warning
                if batch_size > 1:
                    metrics["timestep_std"] = timesteps.float().std().item()
                else:
                    metrics["timestep_std"] = 0.0  # or None if you prefer
                
                return {
                    "loss": loss,
                    "metrics": metrics
                }
                
        except Exception as e:
            logger.error(f"DDPM training step failed: {str(e)}", exc_info=True)
            raise

    def _get_add_time_ids(
        self,
        original_sizes,
        crops_coords_top_left,
        target_size,
        dtype,
        batch_size: int
    ) -> torch.Tensor:
        """Create time embeddings for SDXL."""
        add_time_ids = []
        for i in range(batch_size):
            original_size = original_sizes[i]
            crop_coords = crops_coords_top_left[i]
            
            add_time_id = torch.tensor([
                original_size[0],  # Original image height
                original_size[1],  # Original image width
                crop_coords[0],    # Top coordinate of crop
                crop_coords[1],    # Left coordinate of crop
                target_size[0],    # Target image height
                target_size[1],    # Target image width
            ])
            add_time_ids.append(add_time_id)
        
        return torch.stack(add_time_ids).to(dtype=dtype) 