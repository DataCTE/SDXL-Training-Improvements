"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process

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
        self.noise_scheduler = model.noise_scheduler
        
        # Get model dtype from parameters and ensure VAE matches
        model_dtype = next(self.model.parameters()).dtype
        self.model.vae.to(dtype=model_dtype)
        
    def train(self, num_epochs: int):
        """Execute training loop for specified number of epochs."""
        total_steps = len(self.train_dataloader) * num_epochs
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc=f"Training DDPM ({self.config.training.prediction_type})",
            position=0,
            leave=True,
            ncols=100
        )
        
        try:
            if is_main_process() and self.wandb_logger:
                # Log initial model configuration
                self.wandb_logger.log_model(
                    model=self.model,
                    optimizer=self.optimizer
                )
                
                # Log training configuration
                self.wandb_logger.log_hyperparams({
                    'num_epochs': num_epochs,
                    'total_steps': total_steps,
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.optimizer.learning_rate,
                    'prediction_type': self.config.training.prediction_type,
                    'num_timesteps': self.noise_scheduler.config.num_train_timesteps
                })
            
            for epoch in range(num_epochs):
                self.model.train()
                epoch_loss = 0.0
                num_steps = len(self.train_dataloader)
                epoch_metrics = defaultdict(float)
                
                for step, batch in enumerate(self.train_dataloader):
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Execute training step
                    step_output = self.training_step(batch)
                    loss = step_output["loss"]
                    metrics = step_output["metrics"]
                    
                    # Update epoch metrics
                    for k, v in metrics.items():
                        epoch_metrics[k] += v
                    
                    # Backward pass and optimization
                    loss.backward()
                    if self.config.training.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                        metrics['grad_norm'] = grad_norm.item()
                    self.optimizer.step()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    current_step = epoch * num_steps + step + 1
                    
                    if is_main_process():
                        # Update progress bar with more detailed stats
                        progress_bar.set_postfix(
                            {
                                'epoch': f"{epoch + 1}/{num_epochs}",
                                'step': f"{step + 1}/{num_steps}",
                                'loss': f"{loss.item():.4f}",
                                'avg_loss': f"{epoch_loss / (step + 1):.4f}",
                                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                'pred_type': self.config.training.prediction_type
                            },
                            refresh=True
                        )
                        progress_bar.update(1)
                        
                        # Log step metrics
                        if self.wandb_logger:
                            self.wandb_logger.log_metrics({
                                'epoch': epoch + 1,
                                'step': current_step,
                                'global_step': current_step,
                                'learning_rate': self.optimizer.param_groups[0]['lr'],
                                **metrics,
                                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,
                                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2
                            })
                        
                        # Print periodic updates to console
                        if step % max(1, num_steps // 10) == 0:
                            logger.info(
                                f"Epoch {epoch + 1}/{num_epochs} "
                                f"[{step + 1}/{num_steps}] "
                                f"Loss: {loss.item():.4f} "
                                f"Avg Loss: {epoch_loss / (step + 1):.4f} "
                                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                            )
                
                # Log epoch metrics
                avg_epoch_loss = epoch_loss / num_steps
                avg_epoch_metrics = {k: v / num_steps for k, v in epoch_metrics.items()}
                
                if is_main_process():
                    logger.info(
                        f"\nEpoch {epoch + 1}/{num_epochs} completed - "
                        f"Average Loss: {avg_epoch_loss:.6f} "
                        f"Average Timestep: {avg_epoch_metrics.get('timestep_mean', 0):.1f}\n"
                    )
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({
                            'epoch': epoch + 1,
                            'epoch_loss': avg_epoch_loss,
                            'epoch_learning_rate': self.optimizer.param_groups[0]['lr'],
                            **{f"epoch_{k}": v for k, v in avg_epoch_metrics.items()}
                        })
        
        except Exception as e:
            logger.error(
                "Training loop failed",
                exc_info=True,
                extra={
                    'epoch': epoch if 'epoch' in locals() else None,
                    'step': step if 'step' in locals() else None,
                    'error': str(e)
                }
            )
            raise
        finally:
            progress_bar.close()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single training step."""
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

            # Prepare time embeddings for SDXL
            batch_size = latents.shape[0]
            time_ids = self._get_add_time_ids(
                original_sizes=original_sizes,
                crops_coords_top_left=crop_top_lefts,
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
                "timestep_std": timesteps.float().std().item()
            }

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