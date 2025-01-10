from typing import Optional
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.core.logging import UnifiedLogger, LogConfig, WandbLogger
from src.data.config import Config
from src.training.trainers.base_router import BaseTrainer
from src.core.distributed import is_main_process
from src.core.types import ModelWeightDtypes, DataType

logger = UnifiedLogger(LogConfig(name=__name__))

class SDXLTrainer(BaseTrainer):
    """SDXL-specific trainer that handles model saving and training method delegation."""
    
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader: DataLoader,
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
        
        # Get gradient accumulation steps from config
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        
        # Handle optimizer-specific dtype conversions
        if config.optimizer.optimizer_type == "adamw_bf16":
            logger.info("Converting model components to bfloat16 format")
            
            # Create bfloat16 weight configuration
            bfloat16_weights = ModelWeightDtypes.from_single_dtype(DataType.BFLOAT_16)
            
            # Convert model components and ensure they're on GPU
            try:
                logger.info("Moving model components to device and warming up VRAM...")
                
                # UNet is always trained
                model_dtype = bfloat16_weights.unet.to_torch_dtype()
                self.model.unet.to(device=self.device)
                self.model.unet.to(model_dtype)
                
                # Warmup UNet by running a forward pass with matching dtypes
                dummy_latents = torch.randn(1, 4, 128, 128, device=self.device, dtype=model_dtype)
                dummy_timesteps = torch.ones(1, device=self.device, dtype=model_dtype)
                dummy_encoder_hidden_states = torch.randn(1, 77, 2048, device=self.device, dtype=model_dtype)
                dummy_added_cond_kwargs = {
                    "text_embeds": torch.randn(1, 1280, device=self.device, dtype=model_dtype),
                    "time_ids": torch.randn(1, 6, device=self.device, dtype=model_dtype)
                }
                _ = self.model.unet(
                    dummy_latents,
                    dummy_timesteps,
                    encoder_hidden_states=dummy_encoder_hidden_states,
                    added_cond_kwargs=dummy_added_cond_kwargs
                )
                torch.cuda.synchronize()
                logger.info("Converted UNet to bfloat16 and warmed up")
                
                # VAE stays in float16 for stability
                if hasattr(self.model, 'vae'):
                    vae_dtype = torch.float16
                    self.model.vae.to(device=self.device)
                    self.model.vae.to(vae_dtype)
                    # Warmup VAE with float16 - use RGB input (3 channels)
                    dummy_images = torch.randn(1, 3, 1024, 1024, device=self.device, dtype=vae_dtype)  # RGB input
                    _ = self.model.vae.encode(dummy_images)
                    torch.cuda.synchronize()
                    logger.info("VAE moved to device and warmed up")
                
                # Text encoders
                if hasattr(self.model, 'text_encoder_1'):
                    text_encoder_dtype = bfloat16_weights.text_encoder.to_torch_dtype()
                    self.model.text_encoder_1.to(device=self.device)
                    self.model.text_encoder_1.to(text_encoder_dtype)
                    # Warmup text encoder 1
                    dummy_text = torch.ones((1, 77), dtype=torch.long, device=self.device)  # Input IDs stay as long
                    _ = self.model.text_encoder_1(dummy_text)
                    torch.cuda.synchronize()
                    logger.info("Converted text_encoder_1 to bfloat16 and warmed up")
                    
                if hasattr(self.model, 'text_encoder_2'):
                    text_encoder_2_dtype = bfloat16_weights.text_encoder_2.to_torch_dtype()
                    self.model.text_encoder_2.to(device=self.device)
                    self.model.text_encoder_2.to(text_encoder_2_dtype)
                    # Warmup text encoder 2
                    _ = self.model.text_encoder_2(dummy_text)  # Reuse dummy_text since format is same
                    torch.cuda.synchronize()
                    logger.info("Converted text_encoder_2 to bfloat16 and warmed up")
                
                model_dtype = torch.bfloat16
                
                # Clear any remaining memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Log memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)    # Convert to GB
                    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
                    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_memory:.2f}GB peak")
                
            except Exception as e:
                logger.error(f"Failed to convert model components: {e}")
                raise
                
        else:
            model_dtype = next(self.model.parameters()).dtype
        
        logger.info(f"Model components using dtype: {model_dtype}")
        
        # Initialize specific training method through composition
        if config.training.method.lower() == "ddpm":
            from .methods.ddpm_trainer import DDPMTrainer
            self.trainer = DDPMTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                parent_trainer=self,
                **kwargs
            )
        elif config.training.method.lower() == "flow_matching":
            from .methods.flow_matching_trainer import FlowMatchingTrainer
            self.trainer = FlowMatchingTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                device=device,
                wandb_logger=wandb_logger,
                config=config,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported training method: {config.training.method}")
    
    def train(self, num_epochs: int):
        """Delegate training to the specific trainer implementation."""
        if hasattr(self, 'trainer'):
            logger.info(f"Starting training with {self.trainer.__class__.__name__}")
            return self.trainer.train(num_epochs)
        else:
            raise ValueError("No training method initialized")
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save checkpoint in diffusers format using save_pretrained with safetensors."""
        if not is_main_process():
            return
            
        from src.data.utils.paths import convert_windows_path
        
        # Create outputs directory structure
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Create checkpoint directory name
        if is_final:
            save_dir = outputs_dir / "final_checkpoint"
        else:
            save_dir = outputs_dir / f"checkpoint-{epoch:04d}"
        
        # Convert and create path
        save_path = Path(convert_windows_path(str(save_dir)))
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model weights in safetensors format
            logger.info(f"Saving model checkpoint to {save_path}")
            self.model.save_pretrained(
                str(save_path),
                safe_serialization=True
            )
            
            # Save optimizer state
            optimizer_path = save_path / "optimizer.pt"
            logger.info(f"Saving optimizer state to {optimizer_path}")
            torch.save(
                self.optimizer.state_dict(),
                str(optimizer_path),
                _use_new_zipfile_serialization=False
            )
            
            # Save config as JSON
            config_path = save_path / "config.json"
            logger.info(f"Saving training config to {config_path}")
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info(f"Successfully saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
            raise
        
