"""SDXL trainer implementation with extreme memory optimizations."""
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import time

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from src.core.distributed import is_main_process
from src.data.utils.paths import convert_windows_path
from .base import BaseTrainer

logger = get_logger(__name__)

class SDXLTrainer(BaseTrainer):
    """SDXL trainer with memory optimizations and WSL support."""
    
    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        **kwargs
    ):
        super().__init__(model, optimizer, train_dataloader, **kwargs)
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Execute single training step."""
        try:
            # Get model input and conditioning
            pixel_values = batch["pixel_values"].to(self.device)
            prompt_embeds = batch["prompt_embeds"].to(self.device)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device)
            
            # Convert pixel values to latents using VAE
            with torch.no_grad():
                latents = self.model.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.model.vae.config.scaling_factor
            
            # Apply tag weights if available
            if "tag_weights" in batch:
                tag_weights = batch["tag_weights"].to(self.device)
                latents = latents * tag_weights.view(-1, 1, 1, 1)
            
            # Get timesteps using the base trainer's method
            timesteps = self.get_timestep(batch_size=latents.shape[0])
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Convert inputs to model's dtype
            model_dtype = next(self.model.unet.parameters()).dtype
            noisy_latents = noisy_latents.to(dtype=model_dtype)
            timesteps = timesteps.to(dtype=model_dtype)
            prompt_embeds = prompt_embeds.to(dtype=model_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)
            
            # Get model prediction
            model_pred = self.model.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds}
            ).sample
            
            # Calculate loss (ensure both tensors are in float32 for loss calculation)
            loss = torch.nn.functional.mse_loss(
                model_pred.float(),
                noise.float(),
                reduction="mean"
            )
            
            # Log metrics
            metrics = {
                "loss": loss.detach().item(),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            
            if self.wandb_logger and is_main_process():
                self.wandb_logger.log_metrics(metrics)
            
            return {
                "loss": loss,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {str(e)}", exc_info=True)
            raise

    def save_checkpoint(self, path: str, **kwargs):
        """Save training checkpoint with proper path handling."""
        try:
            save_path = convert_windows_path(path)
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "config": self.config.to_dict(),
                "timestamp": time.time()
            }
            
            # Add any additional data
            checkpoint.update(kwargs)
            
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise

    def load_checkpoint(self, path: str):
        """Load training checkpoint with proper path handling."""
        try:
            load_path = convert_windows_path(path)
            if not Path(load_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {load_path}")
                
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            
            logger.info(f"Loaded checkpoint from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
            raise
