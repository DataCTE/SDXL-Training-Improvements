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
import numpy as np

from src.core.logging import UnifiedLogger, LogConfig
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.training.schedulers.novelai_v3 import configure_noise_scheduler
from src.data.dataset import BucketBatchSampler

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
        
        # Create a new dataloader with bucket-aware batching
        dataset = train_dataloader.dataset
        
        # Ensure cache manager is initialized in main process
        if dataset.cache_manager is None:
            from src.data.preprocessing import CacheManager
            dataset.cache_manager = CacheManager(
                cache_dir=config.global_config.cache.cache_dir,
                config=config,
                max_cache_size=config.global_config.cache.max_cache_size,
                device=device
            )
            
            # Load tag index to ensure it's available
            tag_index = dataset.cache_manager.load_tag_index()
            if tag_index is None:
                logger.warning("No tag index found in cache")
        
        # Get dataloader kwargs from config
        dataloader_kwargs = config.training.dataloader_kwargs
        
        # Create a bucket sampler
        bucket_sampler = BucketBatchSampler(
            dataset.bucket_indices,
            batch_size=config.training.batch_size,
            drop_last=True,
            shuffle=dataset.is_train
        )
        
        # Create dataloader with bucket sampler
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=bucket_sampler,  # Use bucket sampler instead of batch_size
            num_workers=dataloader_kwargs["num_workers"],
            pin_memory=dataloader_kwargs["pin_memory"],
            worker_init_fn=self._worker_init_fn
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = configure_noise_scheduler(config, device)
    
    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Initialize worker process."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
            
        dataset = worker_info.dataset
        
        # Set worker device
        if torch.cuda.is_available():
            device_id = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            dataset.device = torch.device(f'cuda:{device_id}')
        else:
            dataset.device = torch.device('cpu')
            
        # Initialize cache manager for worker
        from src.data.preprocessing import CacheManager
        dataset.cache_manager = CacheManager(
            cache_dir=dataset.config.global_config.cache.cache_dir,
            config=dataset.config,
            max_cache_size=dataset.config.global_config.cache.max_cache_size,
            device=dataset.device
        )
    
    def train(self, num_epochs: int):
        """Execute training loop with proper gradient accumulation."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting training with {total_steps} total steps ({num_epochs} epochs)")
        
        # Initialize progress tracking
        global_step = 0
        best_loss = float('inf')
        progress = logger.start_progress(
            total=total_steps,
            desc=f"Training DDPM ({self.config.training.prediction_type})"
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
                        
                        # Update progress with metrics
                        progress.update(1, {
                            'Loss': f"{loss.item():.4f}",
                            'Epoch': f"{epoch + 1}/{num_epochs}",
                            'Step': f"{step}/{len(self.train_dataloader)}",
                            'Time': f"{step_time:.1f}s"
                        })
                        
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
                            progress.update(1)
                        
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
            progress.close()
            
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
                # Extract all tag weights from each image's tag info
                batch_weights = []
                for m in batch["metadata"]:
                    tag_info = m["tag_info"]
                    image_weights = []
                    for tag_type, tags in tag_info["tags"].items():
                        for tag_data in tags:
                            image_weights.append(tag_data["weight"])
                    # Use mean of all tag weights for this image if any exist
                    if image_weights:
                        batch_weights.append(sum(image_weights) / len(image_weights))
                    else:
                        logger.warning("Image found with no tag weights, skipping weight application")
                        batch_weights = None
                        break
                
                # Apply weights if we have them for all images
                if batch_weights:
                    tag_weights = torch.tensor(batch_weights, device=self.device, dtype=model_dtype)
                    loss = loss * tag_weights.mean()
                    
                    # Log weight statistics for debugging
                    if self.wandb_logger:
                        self.wandb_logger.log({
                            "tag_weights/mean": tag_weights.mean().item(),
                            "tag_weights/std": tag_weights.std().item() if len(tag_weights) > 1 else 0.0,
                            "tag_weights/min": tag_weights.min().item(),
                            "tag_weights/max": tag_weights.max().item()
                        })

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

    
