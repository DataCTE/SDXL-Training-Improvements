"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any, Optional, List, Tuple
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import time
import json
from pathlib import Path
import multiprocessing
import threading

from src.core.logging import UnifiedLogger, LogConfig
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.training.schedulers.novelai_v3 import configure_noise_scheduler

logger = UnifiedLogger(LogConfig(name=__name__))

class DDPMTrainer:
    """DDPM-specific trainer implementation."""
    
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        wandb_logger=None,
        config: Optional[Config] = None,
        parent_trainer=None,
        **kwargs
    ):
        # Store parent trainer reference
        self.parent_trainer = parent_trainer
        
        # Direct initialization
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.wandb_logger = wandb_logger
        self.config = config
        
        # Create a new dataloader with proper multiprocessing settings
        dataset = train_dataloader.dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_dataloader.batch_size,
            shuffle=train_dataloader.dataset.is_train,
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
        
        # Log CUDA worker info
        if torch.cuda.is_available() and config.training.num_workers > 0:
            logger.info(
                "Using spawn method for DataLoader workers with CUDA. "
                "This may have a small startup overhead but is required for stability."
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
        
        # Initialize noise scheduler
        self.noise_scheduler = configure_noise_scheduler(config, device)
        
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
                            # Call parent's save_checkpoint through the trainer reference
                            self.parent_trainer.save_checkpoint(epoch + 1, is_final=False)
                            logger.info(f"Saved checkpoint for epoch {epoch + 1} with loss {best_loss:.4f}")
                    
                    # Clear CUDA cache between epochs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
        except Exception as e:
            logger.error(f"Training loop failed: {str(e)}", exc_info=True)
            raise
        finally:
            progress_bar.close()
            
            # Save final checkpoint through parent trainer
            if is_main_process():
                self.parent_trainer.save_checkpoint(num_epochs, is_final=True)
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
        """Execute single training step with proper tensor handling."""
        try:
            # Validate batch contents
            required_keys = {
                "vae_latents", "prompt_embeds", "pooled_prompt_embeds", 
                "time_ids", "metadata"
            }
            if not all(k in batch for k in required_keys):
                missing = required_keys - set(batch.keys())
                raise ValueError(f"Batch missing required keys: {missing}")

            # Extract tensors and ensure proper device/dtype
            vae_latents = batch["vae_latents"].to(self.device)  # [B, C, H, W]
            prompt_embeds = batch["prompt_embeds"].to(self.device)  # [B, 77, 2048]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device)  # [B, 1, 1280]
            time_ids = batch["time_ids"].to(self.device)  # [B, 1, 6]
            
            # Get batch size for metrics
            batch_size = vae_latents.shape[0]
            model_dtype = vae_latents.dtype

            # Sample noise and timesteps
            noise = torch.randn_like(vae_latents)
            timesteps = self.noise_scheduler.sample_timesteps(batch_size, device=self.device)

            # Add noise according to timesteps
            noisy_latents = self.noise_scheduler.add_noise(
                vae_latents, 
                noise, 
                timesteps
            )

            # Prepare conditioning
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": time_ids
            }

            # Get model prediction
            model_pred = self.model.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # Get target based on prediction type
            if self.config.training.prediction_type == "epsilon":
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(vae_latents, noise, timesteps)
            else:
                target = noise

            # Apply MinSNR loss weighting if enabled
            if self.config.model.min_snr_gamma is not None:
                snr = self.noise_scheduler.get_snr(timesteps)
                mse = F.mse_loss(model_pred, target, reduction="none")
                loss = mse * torch.minimum(
                    snr,
                    torch.ones_like(snr) * self.config.model.min_snr_gamma
                ).float()
                loss = loss.mean()
            else:
                loss = F.mse_loss(model_pred, target)

            # Apply tag weights from metadata if present
            if "tag_info" in batch["metadata"]:
                tag_weights = torch.tensor(
                    [m["tag_info"]["total_weight"] for m in batch["metadata"]], 
                    device=self.device, 
                    dtype=model_dtype
                )
                loss = loss * tag_weights.mean()

            # Ensure loss is finite and properly scaled
            if not torch.isfinite(loss):
                logger.warning("Invalid loss detected - using fallback value")
                loss = torch.tensor(1000.0, device=self.device, dtype=model_dtype)
            else:
                loss = torch.clamp(loss, max=1000.0)

            metrics = {
                "loss": loss.detach().item(),
                "lr": self.optimizer.param_groups[0]["lr"],
                "timestep_mean": timesteps.float().mean().item(),
                "noise_scale": noise.abs().mean().item(),
                "pred_scale": model_pred.abs().mean().item(),
                "batch_size": batch_size
            }

            if batch_size > 1:
                metrics["timestep_std"] = timesteps.float().std().item()

            return {
                "loss": loss,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"DDPM training step failed: {str(e)}", exc_info=True)
            raise

    
