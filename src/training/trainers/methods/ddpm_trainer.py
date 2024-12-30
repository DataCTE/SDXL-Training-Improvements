"""DDPM trainer implementation with memory optimizations."""
import torch
from typing import Dict, Any
import torch.nn.functional as F
from tqdm import tqdm

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
        
    def train(self, num_epochs: int):
        """Execute training loop for specified number of epochs."""
        total_steps = len(self.train_dataloader) * num_epochs
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc="Training"
        )
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_steps = len(self.train_dataloader)
            
            for step, batch in enumerate(self.train_dataloader):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Execute training step
                step_output = self.training_step(batch)
                loss = step_output["loss"]
                metrics = step_output["metrics"]
                
                # Backward pass and optimization
                loss.backward()
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                self.optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                if is_main_process():
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'loss': loss.item(),
                        'avg_loss': epoch_loss / (step + 1)
                    })
                    
                    # Log step metrics
                    if self.wandb_logger:
                        self.wandb_logger.log_metrics({
                            'epoch': epoch + 1,
                            'step': step + 1,
                            **metrics
                        })
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / num_steps
            if is_main_process():
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")
                if self.wandb_logger:
                    self.wandb_logger.log_metrics({
                        'epoch': epoch + 1,
                        'epoch_loss': avg_epoch_loss
                    })
        
        progress_bar.close()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single training step."""
        try:
            # Get batch data and ensure consistent dtype
            latents = batch["pixel_values"].to(self.device, dtype=self.model.dtype)
            prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=self.model.dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=self.model.dtype)

            # Sample noise and timesteps
            noise = torch.randn_like(latents, device=self.device, dtype=self.model.dtype)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=self.device
            ).long()

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps
            )

            # Get model prediction
            model_pred = self.model.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds}
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
                model_pred.to(dtype=self.model.dtype),
                target.to(dtype=self.model.dtype),
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