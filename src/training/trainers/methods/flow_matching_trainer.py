"""Flow Matching trainer implementation with extreme speedups."""
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from torch import Tensor
from tqdm import tqdm

from src.core.logging import get_logger, MetricsLogger
from src.training.trainers.sdxl_trainer import SDXLTrainer
from src.training.schedulers import get_add_time_ids
from src.data.config import Config
from src.core.distributed import is_main_process
from src.core.logging import WandbLogger

logger = get_logger(__name__)

class FlowMatchingTrainer(SDXLTrainer):
    """Flow Matching trainer with memory optimizations."""
    
    name = "flow_matching"

    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        wandb_logger: Optional[WandbLogger] = None,
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
        self.metrics_logger = MetricsLogger(window_size=100)
        self._shape_logs = []
        
        # Try to compile loss computation if available
        if hasattr(torch, "compile"):
            self.logger.debug("Attempting to compile loss computation")
            try:
                self._compiled_loss = torch.compile(
                    self._compute_loss_impl,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                self.logger.debug("Loss computation successfully compiled")
            except Exception as e:
                self.logger.warning(
                    "Failed to compile loss computation",
                    exc_info=True,
                    extra={'error': str(e)}
                )

    def train(self, num_epochs: int):
        """Execute training loop for specified number of epochs."""
        total_steps = len(self.train_dataloader) * num_epochs
        progress_bar = tqdm(
            total=total_steps,
            disable=not is_main_process(),
            desc="Training Flow Matching"
        )
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_steps = len(self.train_dataloader)
            
            for step, batch in enumerate(self.train_dataloader):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute loss
                loss_output = self.compute_loss(self.model, batch)
                loss = loss_output["loss"]
                metrics = loss_output["metrics"]
                
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
                        'avg_loss': epoch_loss / (step + 1),
                        'x0_norm': metrics['x0_norm'],
                        'x1_norm': metrics['x1_norm']
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

    def compute_loss(self, model, batch, generator=None) -> Dict[str, Tensor]:
        """Compute training loss."""
        if hasattr(self, '_compiled_loss'):
            return self._compiled_loss(model, batch, generator)
        return self._compute_loss_impl(model, batch, generator)

    def _compute_loss_impl(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, Tensor]:
        """Compute Flow Matching loss with new data format."""
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

            # Use latents as target (x1)
            x1 = latents

            # Sample time steps
            t = self.sample_logit_normal(
                (x1.shape[0],),
                x1.device,
                x1.dtype,
                generator=generator
            )

            # Generate random starting point with matching dtype
            x0 = torch.randn_like(x1, device=self.device, dtype=model_dtype)

            # Prepare time embeddings for SDXL
            time_ids = self._get_add_time_ids(
                original_sizes=original_sizes,
                crops_coords_top_left=crop_top_lefts,
                target_size=(pixel_values.shape[-2], pixel_values.shape[-1]),
                dtype=prompt_embeds.dtype,
                batch_size=x1.shape[0]
            )
            time_ids = time_ids.to(device=self.device)

            # Prepare conditioning embeddings
            cond_emb = {
                "prompt_embeds": prompt_embeds,
                "added_cond_kwargs": {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids
                }
            }

            # Compute flow matching loss
            loss = self.compute_flow_matching_loss(self.model.unet, x0, x1, t, cond_emb)
            loss = loss.mean()

            # Compute additional metrics
            metrics = {
                "loss": loss.detach().item(),
                "x0_norm": x0.norm().item(),
                "x1_norm": x1.norm().item(),
                "time_mean": t.mean().item(),
                "time_std": t.std().item()
            }

            return {
                "loss": loss,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(
                "Error computing Flow Matching loss",
                exc_info=True,
                extra={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                    'device_info': {
                        'x1_device': x1.device if 'x1' in locals() else None,
                        'prompt_device': prompt_embeds.device if 'prompt_embeds' in locals() else None
                    }
                }
            )
            raise

    def _log_tensor_shapes(self, tensors: Dict[str, Tensor], step: str):
        """Log tensor shapes for debugging."""
        shapes = {
            name: tensor.shape if hasattr(tensor, 'shape') else None 
            for name, tensor in tensors.items()
        }
        self._shape_logs.append({
            'step': step,
            'shapes': shapes,
            'devices': {
                name: str(tensor.device) if hasattr(tensor, 'device') else None
                for name, tensor in tensors.items()
            }
        })

    def sample_logit_normal(
        self, 
        shape, 
        device, 
        dtype, 
        mean=0.0, 
        std=1.0, 
        generator=None
    ) -> Tensor:
        """Sample from logit-normal distribution."""
        normal = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        normal = mean + std * normal
        return torch.sigmoid(normal)

    def optimal_transport_path(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Compute optimal transport path."""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x0 + t * x1

    def compute_velocity(
        self, 
        model: torch.nn.Module, 
        xt: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute velocity field."""
        return model(
            xt,
            t,
            encoder_hidden_states=cond_emb["prompt_embeds"],
            added_cond_kwargs=cond_emb["added_cond_kwargs"]
        ).sample

    def compute_flow_matching_loss(
        self, 
        model: torch.nn.Module, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        cond_emb: Dict[str, Any]
    ) -> Tensor:
        """Compute flow matching loss."""
        xt = self.optimal_transport_path(x0, x1, t)
        v_true = x1 - x0
        v_pred = self.compute_velocity(model, xt, t, cond_emb)
        return F.mse_loss(v_pred, v_true, reduction="none").mean([1, 2, 3])
