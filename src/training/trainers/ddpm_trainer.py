"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any
import torch.nn.functional as F

from src.core.logging import get_logger
from src.models import StableDiffusionXL
from .sdxl_trainer import SDXLTrainer
from src.core.distributed import is_main_process

logger = get_logger(__name__)

class DDPMTrainer(SDXLTrainer):
    """DDPM-specific trainer implementation."""
    
    def __init__(
        self,
        model: StableDiffusionXL,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        **kwargs
    ):
        super().__init__(model, optimizer, train_dataloader, **kwargs)
        
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Execute single DDPM training step."""
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
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=latents.device
            )
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get model prediction
            model_pred = self.model.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds}
            ).sample
            
            # Calculate loss
            if self.config.training.prediction_type == "epsilon":
                target = noise
            elif self.config.training.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.config.training.prediction_type}")
                
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
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