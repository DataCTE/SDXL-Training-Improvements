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
        
        # Initialize cache manager for tag weights
        from src.data.preprocessing.cache_manager import CacheManager
        self.cache_manager = CacheManager(
            cache_dir=config.global_config.cache.cache_dir,
            device=device
        )
        
        # Load tag weights from cache
        self.tag_weights_cache = {}
        self.tag_weights_path = self.cache_manager.cache_dir / "tag_weights.json"
        self.tag_index_path = self.cache_manager.cache_dir / "tag_weights_index.json"
        
        if not (self.tag_weights_path.exists() and self.tag_index_path.exists()):
            raise ValueError(
                f"Tag weights files not found at {self.tag_weights_path} or {self.tag_index_path}. "
                "Please run preprocessing first."
            )
        
        try:
            # Load both JSON files
            with open(self.tag_weights_path, 'r') as f:
                self.tag_weights_data = json.load(f)
            with open(self.tag_index_path, 'r') as f:
                self.tag_index_data = json.load(f)
            logger.info("Loaded tag weights from cache")
            
            # Pre-cache metadata for faster lookup
            self.metadata = self.tag_index_data["metadata"]
            self.default_weight = float(self.metadata["default_weight"])
            
        except Exception as e:
            logger.error(f"Failed to load tag weights cache: {e}")
            raise
        
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
        
    def get_tag_weights(self, texts: List[str]) -> torch.Tensor:
        """Get cached tag weights for a batch of texts efficiently."""
        weights = []
        for text in texts:
            # First check memory cache
            if text in self.tag_weights_cache:
                weights.append(self.tag_weights_cache[text])
                continue
            
            # Then check index data
            image_data = self.tag_index_data["images"].get(text, None)
            if image_data:
                weight = float(image_data["total_weight"])
            else:
                # If not found, calculate from tag weights
                try:
                    tags = text.split(",")  # Split caption into tags
                    tag_weights = []
                    for tag in tags:
                        tag = tag.strip().lower()
                        # Check all tag types from metadata
                        for tag_type in ["subject", "style", "quality", "technical", "meta"]:
                            if tag in self.tag_weights_data["tag_weights"][tag_type]:
                                tag_weights.append(
                                    self.tag_weights_data["tag_weights"][tag_type][tag]
                                )
                                break
                
                    # Calculate geometric mean of tag weights
                    if tag_weights:
                        import numpy as np
                        weight = float(np.exp(np.mean(np.log(tag_weights))))
                    else:
                        weight = self.default_weight
                        
                except Exception as e:
                    logger.warning(f"Error calculating weight for text: {text[:50]}... - {str(e)}")
                    weight = self.default_weight
            
            # Cache the result
            self.tag_weights_cache[text] = weight
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train(self, num_epochs: int):
        """Execute training loop with proper gradient accumulation."""
        total_steps = len(self.train_dataloader) * num_epochs
        logger.info(f"Starting training with {total_steps} total steps ({num_epochs} epochs)")
        
        # Initialize progress tracking
        global_step = 0
        current_loss = float('inf')  # Initialize loss tracking
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
                
                accumulated_loss = 0.0
                accumulated_metrics = defaultdict(float)
                
                for step, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    try:
                        # Execute training step and accumulate gradients
                        loss, metrics = self._execute_training_step(
                            batch, 
                            accumulate=True,
                            is_last_accumulation_step=((step + 1) % self.gradient_accumulation_steps == 0)
                        )
                        
                        # Skip this batch if loss is invalid
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Skipping step {step} due to invalid loss")
                            continue
                            
                        step_time = time.time() - step_start_time
                        
                        # Update current loss tracking
                        current_loss = loss.item()
                        
                        # Update progress bar
                        progress_bar.set_postfix(
                            {'Loss': f"{current_loss:.4f}", 'Time': f"{step_time:.1f}s"},
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
                        
                    except ValueError as e:
                        logger.warning(f"Skipping batch due to error: {str(e)}")
                        continue
                    
                    except Exception as e:
                        logger.error(f"Unexpected error during training step: {str(e)}")
                        raise
                
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
            
            # Generate cache key from image path
            cache_key = self.cache_manager.get_cache_key(batch["image_path"])
            
            # Load all cached data at once
            cached_data = self.cache_manager.load_tensors(cache_key)
            if cached_data is None:
                raise ValueError(f"No cached data found for key: {cache_key}")
            
            # Extract all cached tensors and move to device with correct dtype
            latents = cached_data["pixel_values"].to(device=self.device, dtype=model_dtype)
            prompt_embeds = cached_data["prompt_embeds"].to(device=self.device, dtype=model_dtype)
            pooled_prompt_embeds = cached_data["pooled_prompt_embeds"].to(device=self.device, dtype=model_dtype)
            metadata = cached_data.get("metadata", {})
            
            # Use cached metadata for dimensions and coordinates
            original_size = metadata["original_size"]
            bucket_size = metadata["bucket_size"]
            
            # Use context manager for mixed precision
            with autocast(device_type='cuda', enabled=self.mixed_precision != "no"):
                # Prepare time embeddings using cached metadata
                batch_size = latents.shape[0]
                time_ids = self._get_add_time_ids(
                    original_sizes=[original_size] * batch_size,  # Use cached original size
                    crops_coords_top_left=[(0, 0)] * batch_size,  # Use cached crop coords if available
                    target_size=bucket_size,  # Use cached bucket size
                    dtype=prompt_embeds.dtype,
                    batch_size=batch_size
                )
                time_ids = time_ids.to(device=self.device)

                # Sample noise with proper scaling
                noise = torch.randn_like(latents, device=self.device, dtype=model_dtype)
                noise = torch.clamp(noise, -20000.0, 20000.0)  # Clamp noise values
                
                # Sample timesteps with proper scaling
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), device=self.device
                ).long()

                # Add noise to latents with value checking and scaling
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Clamp noisy latents to prevent extreme values
                noisy_latents = torch.clamp(noisy_latents, -20000.0, 20000.0)
                
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

                # Calculate base loss with consistent dtype and scaling
                if self.config.training.prediction_type == "epsilon":
                    target = noise
                elif self.config.training.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
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
                
                # Get tag weights from cache for the batch
                tag_weights = self.get_tag_weights(batch["text"])
                
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

                # Log metrics with additional noise and weight statistics
                metrics = {
                    "loss": loss.detach().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "timestep_mean": timesteps.float().mean().item(),
                    "pred_max": model_pred.abs().max().item(),
                    "target_max": target.abs().max().item(),
                    "noise_scale": noise.abs().mean().item(),
                    "latent_scale": latents.abs().mean().item(),
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