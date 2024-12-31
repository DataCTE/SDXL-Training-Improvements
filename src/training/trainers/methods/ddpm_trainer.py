"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import time

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process
from src.core.types import DataType, ModelWeightDtypes

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
        config=None,
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
        
        # Enable memory optimizations
        self.model.enable_gradient_checkpointing()
        self.model.enable_xformers_memory_efficient_attention()
        
        # Move model components to device and optimize memory
        self.model.to(device)
        if hasattr(self.model, "enable_model_cpu_offload"):
            self.model.enable_model_cpu_offload()
        
        # Set up automatic mixed precision
        self.mixed_precision = config.training.get("mixed_precision", "no")
        if self.mixed_precision != "no":
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.noise_scheduler = model.noise_scheduler
        
        # Check if using AdamWBF16 optimizer and convert model to bfloat16 if needed
        if config.optimizer.optimizer_type == "adamw_bf16":
            logger.info("Converting model to bfloat16 format for AdamWBF16 optimizer")
            # Create bfloat16 weight configuration
            bfloat16_weights = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
            # Convert model components to bfloat16
            self.model.unet.to(bfloat16_weights.unet.to_torch_dtype())
            self.model.vae.to(bfloat16_weights.vae.to_torch_dtype())
            self.model.text_encoder_1.to(bfloat16_weights.text_encoder.to_torch_dtype())
            self.model.text_encoder_2.to(bfloat16_weights.text_encoder_2.to_torch_dtype())
            model_dtype = torch.bfloat16
        else:
            # Get model dtype from parameters for other optimizers
            model_dtype = next(self.model.parameters()).dtype
        
        logger.info(f"Model components using dtype: {model_dtype}")
        
    def train(self, num_epochs: int):
        """Execute training loop for specified number of epochs."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting training with {total_steps} total steps ({num_epochs} epochs)")
        
        # Initialize progress tracking
        global_step = 0
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc=f"Training DDPM ({self.config.training.prediction_type})",
            position=0,
            leave=True,
            ncols=100,
            dynamic_ncols=True,
            mininterval=0.1,
            smoothing=0.1
        )
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                self.model.train()
                epoch_loss = 0.0
                num_steps = len(self.train_dataloader)
                
                for step, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    global_step = epoch * num_steps + step
                    
                    # Training step
                    loss, metrics = self._execute_training_step(batch)
                    epoch_loss += loss.item()
                    
                    step_time = time.time() - step_start_time
                    
                    if is_main_process():
                        # Log detailed step info periodically
                        if step == 0 or step % 10 == 0:
                            logger.info(
                                f"E{epoch + 1}[{step}/{num_steps}] "
                                f"Loss: {loss.item():.4f} "
                                f"Time: {step_time:.2f}s "
                                f"GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB"
                            )
                        
                        # Update progress bar
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {
                                'E': f"{epoch + 1}/{num_epochs}",
                                'S': f"{step + 1}/{num_steps}",
                                'Loss': f"{loss.item():.4f}",
                                'Time': f"{step_time:.1f}s"
                            },
                            refresh=True
                        )
                        
                        # Log metrics
                        if self.wandb_logger:
                            self.wandb_logger.log_metrics(
                                {
                                    'epoch': epoch + 1,
                                    'step': step + 1,
                                    'global_step': global_step,
                                    'loss': loss.item(),
                                    'step_time': step_time,
                                    **metrics
                                },
                                step=global_step
                            )
                
                # Log epoch summary
                avg_epoch_loss = epoch_loss / num_steps
                logger.info(f"Epoch {epoch + 1} completed - Avg Loss: {avg_epoch_loss:.4f}")
                
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch + 1}, step {step + 1}", exc_info=True)
            raise
        finally:
            progress_bar.close()

    def _execute_training_step(self, batch):
        """Execute single training step with proper error handling."""
        try:
            self.optimizer.zero_grad()
            step_output = self.training_step(batch)
            loss = step_output["loss"]
            metrics = step_output["metrics"]
            
            loss.backward()
            if self.config.training.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.clip_grad_norm
                )
                metrics['grad_norm'] = grad_norm.item()
            
            self.optimizer.step()
            
            return loss, metrics
            
        except Exception as e:
            logger.error("Training step failed", exc_info=True)
            raise

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single training step with memory optimizations."""
        try:
            # Clear cache before forward pass
            torch.cuda.empty_cache()
            
            # Get model dtype from parameters
            model_dtype = next(self.model.parameters()).dtype
            
            # Move batch data to device efficiently
            pixel_values = batch["pixel_values"].to(self.device, dtype=model_dtype, non_blocking=True)
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=model_dtype, non_blocking=True)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=model_dtype, non_blocking=True)
            
            # Use context manager for mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision != "no"):
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
                    target_size=(pixel_values.shape[-2], pixel_values.shape[-1]),  # Use original image dimensions
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
                
                if self.wandb_logger and is_main_process():
                    self.wandb_logger.log_metrics(metrics)

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