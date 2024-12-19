"""SDXL trainer implementation."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

from ..core.memory.tensor import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_gc
)
from ..core.types import DataType

from ..config import Config
from ..core.distributed import is_main_process, get_world_size
from ..core.logging import log_metrics
from ..core.logging.wandb import WandbLogger
from ..models import StableDiffusionXLModel
from .noise import generate_noise, get_add_time_ids

logger = logging.getLogger(__name__)

class SDXLTrainer:
    """Trainer for SDXL fine-tuning with advanced features."""
    
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        scheduler: DDPMScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            accelerator: Accelerator for distributed training
            unet: UNet model
            optimizer: Optimizer
            scheduler: Noise scheduler
            train_dataloader: Training data loader
            device: Target device
        """
        self.config = config
        self.model = model
        self.unet = model.unet  # Extract UNet from SDXL model
        self.optimizer = optimizer
        self.noise_scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.device = device
        self.wandb_logger = wandb_logger
        
        # Initialize validator
        if is_main_process():
            from ..core.validation.text_to_image import TextToImageValidator
            self.validator = TextToImageValidator(
                base_model_path=config.model.pretrained_model_name,
                device=device,
                output_dir=config.global_config.output_dir,
                validation_prompts=validation_prompts
            )
        
        # Move model and optimizer to device efficiently
        if not tensors_match_device(self.model.state_dict(), device):
            with create_stream_context(torch.cuda.current_stream()):
                tensors_to_device_(self.model.state_dict(), device, non_blocking=True)
                if hasattr(self.optimizer, 'state'):
                    tensors_to_device_(self.optimizer.state, device, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            torch_gc()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.max_steps = config.training.max_train_steps or (
            len(train_dataloader) * config.training.num_epochs
        )
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, float]:
        """Execute single training step."""
        if self.config.training.method == "flow_matching":
            return self._train_step_flow_matching(batch)
        else:
            return self._train_step_ddpm(batch, generator)
            
    def _train_step_flow_matching(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute Flow Matching training step."""
        from .flow_matching import sample_logit_normal, compute_flow_matching_loss
        
        # Get batch inputs
        x1 = batch["model_input"]  # Target samples
        
        # Sample time values from logit-normal
        t = sample_logit_normal(
            (x1.shape[0],),
            device=self.device,
            dtype=x1.dtype
        )
        
        # Sample initial points
        x0 = torch.randn_like(x1)
        
        # Get conditioning
        condition_embeddings = {
            "prompt_embeds": batch["prompt_embeds"],
            "added_cond_kwargs": {
                "text_embeds": batch["pooled_prompt_embeds"],
                "time_ids": get_add_time_ids(
                    batch["original_sizes"],
                    batch["crop_top_lefts"],
                    batch["target_sizes"],
                    dtype=batch["prompt_embeds"].dtype,
                    device=self.device
                )
            }
        }
        
        # Compute loss
        loss = compute_flow_matching_loss(
            self.model.unet,
            x0,
            x1,
            t,
            condition_embeddings
        )
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"].mean()
            
        # Backpropagate
        loss.backward()
        
        # Gradient clipping
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(),
                self.config.training.max_grad_norm
            )
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {"loss": loss.detach().item()}
        
    def _train_step_ddpm(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, float]:
        """Execute single training step.
        
        Args:
            batch: Batch of training data
            generator: Optional random number generator
            
        Returns:
            Dict of metrics
        """
        # Get batch inputs
        latents = batch["model_input"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]
        
        # Add noise
        noise = generate_noise(
            latents.shape,
            device=self.device,
            dtype=latents.dtype,
            generator=generator,
            layout=latents
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get time embeddings
        add_time_ids = get_add_time_ids(
            batch["original_sizes"],
            batch["crop_top_lefts"],
            batch["target_sizes"],
            dtype=prompt_embeds.dtype,
            device=self.device
        )
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        ).sample
        
        # Compute loss
        if self.config.training.prediction_type == "epsilon":
            target = noise
        elif self.config.training.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.config.training.prediction_type}")
            
        loss = F.mse_loss(noise_pred, target, reduction="none")
        loss = loss.mean([1, 2, 3])
        
        # Apply loss weights if provided
        if "loss_weights" in batch:
            loss = loss * batch["loss_weights"]
            
        loss = loss.mean()
        
        # Backpropagate
        loss.backward()
        
        # Gradient clipping
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(),
                self.config.training.max_grad_norm
            )
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {"loss": loss.detach().item()}
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict of epoch metrics
        """
        epoch_metrics = {}
        
        # Set models to training mode
        self.model.train()
        
        # Training loop
        progress_bar = tqdm(
            total=len(self.train_dataloader),
            disable=not is_main_process(),
            desc=f"Epoch {self.epoch}"
        )
        
        for batch in self.train_dataloader:
            # Skip if max steps reached
            if self.global_step >= self.max_steps:
                break
                
            # Training step
            step_metrics = self.train_step(batch)
            
            # Update metrics
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
                
            # Log metrics
            if self.global_step % self.config.training.log_steps == 0:
                log_metrics(
                    step_metrics,
                    self.global_step,
                    is_main_process=is_main_process(),
                    use_wandb=False  # Handled by WandbLogger
                )
                
                if self.wandb_logger is not None and is_main_process():
                    self.wandb_logger.log_metrics(
                        step_metrics,
                        step=self.global_step
                    )
                
            # Save checkpoint
            if (
                self.config.training.save_steps > 0 and
                self.global_step % self.config.training.save_steps == 0
            ):
                self.save_checkpoint()
                
            self.global_step += 1
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Compute epoch metrics
        epoch_metrics = {
            k: sum(v) / len(v)
            for k, v in epoch_metrics.items()
        }
        
        return epoch_metrics
        
    def train(self) -> Dict[str, float]:
        """Execute complete training loop.
        
        Returns:
            Dict of final metrics
        """
        logger.info("Starting training...")
        
        metrics = {}
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            log_metrics(
                epoch_metrics,
                epoch,
                is_main_process=is_main_process(),
                use_wandb=self.config.training.use_wandb,
                step_type="epoch"
            )
            
            # Update metrics
            metrics.update(epoch_metrics)
            
            # Run validation
            if (
                is_main_process() and
                self.config.training.validation_steps > 0 and
                self.global_step % self.config.training.validation_steps == 0
            ):
                # Create validation copy of model
                validation_model = self.model.create_validation_copy()
                validation_model.to(self.device)
                
                self.validator.validate(
                    model=validation_model,
                    step=self.global_step,
                    seed=self.config.global_config.seed
                )
                
                del validation_model
                torch.cuda.empty_cache()
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Check if max steps reached
            if self.global_step >= self.max_steps:
                break
                
        return metrics
        
    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        if not is_main_process():
            return
            
        logger.info(f"Saving checkpoint at step {self.global_step}...")
        
        checkpoint_dir = Path(self.config.global_config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch
            },
            checkpoint_dir / "state.pt"
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
