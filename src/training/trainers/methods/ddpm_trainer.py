"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any, Optional, List
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import time
import json
from pathlib import Path

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer, save_checkpoint
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
        
        # Remove tag weight caching since weights now come from dataset
        self.default_weight = config.tag_weighting.default_weight
        
        # Verify effective batch size with fixed gradient accumulation
        self.effective_batch_size = (
            config.training.batch_size * 
            config.training.gradient_accumulation_steps  # Use config value
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
        best_loss = float('inf')
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
                
                epoch_loss = 0.0
                epoch_metrics = defaultdict(float)
                valid_steps = 0
                
                for step, batch in enumerate(self.train_dataloader):
                    try:
                        # Skip invalid batches
                        if batch is None:
                            logger.warning(f"Skipping invalid batch at step {step}")
                            continue
                            
                        step_start_time = time.time()
                        
                        # Execute training step with gradient accumulation
                        loss, metrics = self._execute_training_step(
                            batch, 
                            accumulate=True,
                            is_last_accumulation_step=((step + 1) % self.config.training.gradient_accumulation_steps == 0)
                        )
                        
                        # Skip if loss is invalid
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Skipping step {step} due to invalid loss")
                            continue
                            
                        step_time = time.time() - step_start_time
                        
                        # Update epoch accumulators
                        epoch_loss += loss.item()
                        for k, v in metrics.items():
                            epoch_metrics[k] += v
                        valid_steps += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix(
                            {
                                'Loss': f"{loss.item():.4f}",
                                'Epoch': f"{epoch + 1}/{num_epochs}",
                                'Step': f"{step}/{len(self.train_dataloader)}",
                                'Time': f"{step_time:.1f}s"
                            },
                            refresh=True
                        )
                        
                        # Log step metrics if it's the last accumulation step
                        if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                            if is_main_process() and self.wandb_logger:
                                self.wandb_logger.log_metrics(
                                    {
                                        'epoch': epoch + 1,
                                        'step': global_step,
                                        'loss': loss.item(),
                                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                                        'step_time': step_time,
                                        **metrics
                                    },
                                    step=global_step
                                )
                            
                            global_step += 1
                            progress_bar.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Error in training step {step}: {str(e)}", exc_info=True)
                        continue
                    
               
                # Compute epoch metrics
                if valid_steps > 0:
                    avg_epoch_loss = epoch_loss / valid_steps
                    avg_epoch_metrics = {
                        k: v / valid_steps for k, v in epoch_metrics.items()
                    }
                    
                    # Log epoch metrics
                    if is_main_process():
                        logger.info(
                            f"Epoch {epoch + 1} summary:\n"
                            f"Average loss: {avg_epoch_loss:.4f}\n"
                            f"Metrics: {json.dumps(avg_epoch_metrics, indent=2)}"
                        )
                        
                        if self.wandb_logger:
                            self.wandb_logger.log_metrics(
                                {
                                    'epoch': epoch + 1,
                                    'epoch_loss': avg_epoch_loss,
                                    **{f"epoch_{k}": v for k, v in avg_epoch_metrics.items()}
                                },
                                step=global_step
                            )
                    
                    # Save checkpoint if loss improved
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        if is_main_process():
                            save_checkpoint(
                                model=self.model,
                                optimizer=self.optimizer,
                                epoch=epoch + 1,
                                config=self.config,
                                is_final=False
                            )
                            logger.info(f"Saved checkpoint for epoch {epoch + 1} with loss {best_loss:.4f}")
                    
                    # Clear CUDA cache between epochs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
        except Exception as e:
            logger.error(f"Training loop failed: {str(e)}", exc_info=True)
            raise
        finally:
            progress_bar.close()
            
            # Save final checkpoint
            if is_main_process():
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=num_epochs,
                    config=self.config,
                    is_final=True
                )
                logger.info("Saved final checkpoint")

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
                loss = loss / self.config.training.gradient_accumulation_steps
            
            loss.backward()
            
            # Only return the unscaled loss for logging
            return loss * self.config.training.gradient_accumulation_steps, metrics
            
        except Exception as e:
            logger.error("Training step failed", exc_info=True)
            raise

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single training step with memory optimizations."""
        try:
            # Get model dtype from parameters
            model_dtype = next(self.model.parameters()).dtype
            
            # Use pre-processed VAE latents directly from batch
            vae_latents = batch["vae_latents"].to(device=self.device, dtype=model_dtype)
            prompt_embeds = batch["prompt_embeds"].to(device=self.device, dtype=model_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=self.device, dtype=model_dtype)
            
            # Get tag weights directly from batch
            tag_weights = batch["tag_weights"].to(device=self.device, dtype=model_dtype)
            
            # Get metadata from batch
            original_sizes = batch["original_size"]
            target_size = batch["target_size"][0] if isinstance(batch["target_size"], list) else batch["target_size"]
            crop_coords = batch.get("crop_top_lefts", [(0, 0)] * vae_latents.shape[0])
            
            # Use context manager for mixed precision
            with autocast(device_type='cuda', enabled=self.mixed_precision != "no"):
                # Prepare time embeddings using metadata
                batch_size = vae_latents.shape[0]
                add_time_ids = torch.cat([
                    self.compute_time_ids(
                        original_size=orig_size,
                        crops_coords_top_left=crop_coord,
                        target_size=target_size,
                        device=self.device,
                        dtype=model_dtype
                    ) for orig_size, crop_coord in zip(original_sizes, crop_coords)
                ])
                add_time_ids = add_time_ids.to(device=self.device)

                # Sample noise with proper scaling
                noise = torch.randn_like(vae_latents, device=self.device, dtype=model_dtype)
                noise = torch.clamp(noise, -20000.0, 20000.0)  # Clamp noise values
                
                # Sample timesteps with proper scaling
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), device=self.device
                ).long()

                # Add noise to latents with value checking and scaling
                noisy_latents = self.noise_scheduler.add_noise(vae_latents, noise, timesteps)
                
                # Clamp noisy latents to prevent extreme values
                noisy_latents = torch.clamp(noisy_latents, -20000.0, 20000.0)
                
                # Get model prediction with time embeddings
                model_pred = self.model.unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids
                    }
                ).sample

                # Calculate base loss with consistent dtype and scaling
                if self.config.training.prediction_type == "epsilon":
                    target = noise
                elif self.config.training.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(vae_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {self.config.training.prediction_type}")

                # Ensure both tensors are in the same dtype and properly scaled
                model_pred = model_pred.to(dtype=model_dtype)
                target = target.to(dtype=model_dtype)

                # Clamp predictions and targets to prevent extreme values
                model_pred = torch.clamp(model_pred, -20000.0, 20000.0)
                target = torch.clamp(target, -20000.0, 20000.0)

                # Calculate unweighted loss
                base_loss = F.mse_loss(model_pred, target, reduction="none")  # Keep per-sample losses
                
                # Reshape weights to match loss dimensions
                tag_weights = tag_weights.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
                
                # Apply tag weights to loss
                weighted_loss = base_loss * tag_weights
                
                # Reduce to scalar loss
                loss = weighted_loss.mean()
                
                # Ensure loss is finite and properly scaled
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Invalid loss detected - using fallback value")
                    loss = torch.tensor(1000.0, device=self.device, dtype=model_dtype)
                else:
                    # Clip loss to prevent explosion
                    loss = torch.clamp(loss, max=1000.0)

                # Log metrics with tag weight statistics
                metrics = {
                    "loss": loss.detach().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "timestep_mean": timesteps.float().mean().item(),
                    "pred_max": model_pred.abs().max().item(),
                    "target_max": target.abs().max().item(),
                    "noise_scale": noise.abs().mean().item(),
                    "latent_scale": vae_latents.abs().mean().item(),
                    "weight_mean": tag_weights.mean().item(),
                    "weight_std": tag_weights.std().item(),
                    "weight_min": tag_weights.min().item(),
                    "weight_max": tag_weights.max().item(),
                    "base_loss_mean": base_loss.mean().item(),
                    "weighted_loss_mean": weighted_loss.mean().item(),
                }
                
                # Only calculate std if batch size > 1 to avoid warning
                if batch_size > 1:
                    metrics["timestep_std"] = timesteps.float().std().item()
                else:
                    metrics["timestep_std"] = 0.0
                
                return {
                    "loss": loss,
                    "metrics": metrics
                }
                
        except Exception as e:
            logger.error(f"DDPM training step failed: {str(e)}", exc_info=True)
            raise

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
