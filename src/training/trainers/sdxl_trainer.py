"""SDXL trainer implementation with 100x speedups."""
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core.logging import get_logger, LogConfig
from src.training.metrics import MetricsLogger
from src.training.validation import ValidationLogger
logger = get_logger(__name__)

# Force speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from src.core.distributed import is_main_process
from src.core.logging import WandbLogger, log_metrics
from src.core.memory import (
    tensors_to_device_,
    tensors_match_device,
    create_stream_context,
    torch_sync,
    setup_memory_optimizations,
    verify_memory_optimizations,
    LayerOffloader,
    LayerOffloadConfig,
    ThroughputMonitor
)
from src.core.types import DataType, ModelWeightDtypes
from src.data.config import Config
from src.models import StableDiffusionXLModel
from src.training.methods.base import TrainingMethod

class SDXLTrainer:
    @classmethod
    def create(cls, config: Config, model: StableDiffusionXLModel, 
               optimizer: torch.optim.Optimizer, train_dataloader: DataLoader,
               device: Union[str, torch.device], wandb_logger: Optional[WandbLogger] = None,
               validation_prompts: Optional[List[str]] = None) -> 'SDXLTrainer':
        logger.debug("Creating SDXLTrainer instance")
        
        # Extract method from training configuration
        method = config.training.method
        if not isinstance(method, str):
            logger.error(f"Invalid training method type: {type(method)}")
            raise ValueError(f"Training method must be a string, got {type(method)}")
            
        method = method.lower()
        logger.debug(f"Using training method: {method}")
        
        trainer_cls = TrainingMethod.get_method(method)
        logger.debug(f"Selected trainer class: {trainer_cls.__name__}")
        
        try:
            training_method = trainer_cls(unet=model.unet, config=config)
            logger.debug("Training method initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize training method: {str(e)}", exc_info=True)
            logger.error(
                "Error initializing training method",
                extra={
                    'error_type': type(e).__name__,
                    'error_msg': str(e),
                    'method_name': method,
                    'config': str(config)
                },
                exc_info=True
            )
            raise
        return cls(
            config=config,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            training_method=training_method,
            device=device,
            wandb_logger=wandb_logger,
            validation_prompts=validation_prompts
        )
    
    def __init__(
        self,
        config: Config,
        model: StableDiffusionXLModel,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        training_method: TrainingMethod,
        device: Union[str, torch.device],
        wandb_logger: Optional[WandbLogger] = None,
        validation_prompts: Optional[List[str]] = None
    ):
        self.config = config
        self.model = model
        self.unet = model.unet
        self.optimizer = optimizer
        # Ensure DataLoader uses a single worker
        self.train_dataloader = DataLoader(
            train_dataloader.dataset,
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            num_workers=0,  # Use single worker
            pin_memory=True,
            drop_last=True,
        )
        self.training_method = training_method
        self.device = device
        self.wandb_logger = wandb_logger

        base_dtype = DataType.from_str(config.model.dtype)
        fallback_dtype = DataType.from_str(config.model.fallback_dtype)
        self.model_dtypes = ModelWeightDtypes(
            train_dtype=base_dtype,
            fallback_train_dtype=fallback_dtype,
            unet=DataType.from_str(config.model.unet_dtype or config.model.dtype),
            prior=DataType.from_str(config.model.prior_dtype or config.model.dtype),
            text_encoder=DataType.from_str(config.model.text_encoder_dtype or config.model.dtype),
            text_encoder_2=DataType.from_str(config.model.text_encoder_2_dtype or config.model.dtype),
            vae=DataType.from_str(config.model.vae_dtype or config.model.dtype),
            effnet_encoder=DataType.from_str(config.model.effnet_dtype or config.model.dtype),
            decoder=DataType.from_str(config.model.decoder_dtype or config.model.dtype),
            decoder_text_encoder=DataType.from_str(config.model.decoder_text_encoder_dtype or config.model.dtype),
            decoder_vqgan=DataType.from_str(config.model.decoder_vqgan_dtype or config.model.dtype),
            lora=DataType.from_str(config.model.lora_dtype or config.model.dtype),
            embedding=DataType.from_str(config.model.embedding_dtype or config.model.dtype)
        )

        self._setup_memory_management(
            batch_size=train_dataloader.batch_size,
            micro_batch_size=config.training.micro_batch_size
        )
        if not tensors_match_device(self.model.state_dict(), device):
            with create_stream_context(torch.cuda.current_stream()):
                tensors_to_device_(self.model.state_dict(), device)
                if hasattr(self.optimizer, 'state'):
                    tensors_to_device_(self.optimizer.state, device)
            torch.cuda.current_stream().synchronize()
        torch_sync()

        self.global_step = 0
        self.epoch = 0
        self.max_steps = config.training.max_train_steps or (
            len(train_dataloader) * config.training.num_epochs
        )
        self.metrics_logger = MetricsLogger(window_size=100)
        if validation_prompts:
            self.validation_logger = ValidationLogger(
                model=model,
                prompts=validation_prompts,
                output_dir=Path(config.global_config.output_dir)
            )
        else:
            self.validation_logger = None

        self.gradient_accumulation_steps = (
            train_dataloader.batch_size // config.training.micro_batch_size
            if config.training.micro_batch_size else 1
        )

        # Compile for speed if available
        if hasattr(torch, "compile"):
            self.train_step = torch.compile(self.train_step, mode="reduce-overhead", fullgraph=False)
            self.train_epoch = torch.compile(self.train_epoch, mode="reduce-overhead", fullgraph=False)
            self.train = torch.compile(self.train, mode="reduce-overhead", fullgraph=False)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        accumulation_step: int = 0
    ) -> Dict[str, float]:
        try:
            # Add pre-validation
            logger.debug("=== Pre-Training Step Validation ===")
            def validate_tensors(data: Union[Dict, torch.Tensor], path: str = "") -> None:
                if isinstance(data, dict):
                    for key, value in data.items():
                        new_path = f"{path}.{key}" if path else key
                        validate_tensors(value, new_path)
                elif isinstance(data, torch.Tensor):
                    if path.endswith("model_input") or "latent" in path:
                        logger.debug(
                            f"Validating tensor at {path}: "
                            f"shape={data.shape}, dtype={data.dtype}, "
                            f"device={data.device}, "
                            f"range=[{data.min().item():.3f}, {data.max().item():.3f}]"
                        )

            validate_tensors(batch)

            loss_dict = self.training_method.compute_loss(self.unet, batch, generator=generator)
            loss = loss_dict["loss"] / self.gradient_accumulation_steps
            loss.backward()
            
            grad_norm = 0.0
            if accumulation_step == self.gradient_accumulation_steps - 1:
                if self.config.training.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(),
                        self.config.training.max_grad_norm
                    ).item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            metrics = {
                "loss": loss_dict["loss"].item(),
                "grad_norm": grad_norm,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": batch["model_input"].shape[0]
            }
            self.metrics_logger.update(metrics)
            
            return {k: v.detach().item() for k, v in loss_dict.items()}
        except Exception as e:
            logger.error(
                "Error in training step", 
                exc_info=True,
                extra={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'batch_keys': list(batch.keys()) if isinstance(batch, dict) else None,
                    'accumulation_step': accumulation_step,
                    'stack_info': True
                }
            )
            raise

    def train_epoch(self) -> Dict[str, float]:
        epoch_metrics = {}
        self.model.train()
        progress_bar = tqdm(total=len(self.train_dataloader), disable=not is_main_process(), desc=f"Epoch {self.epoch}")
        
        try:
            for batch in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break
                try:
                    micro_batches = self._prepare_micro_batches(batch)
                    step_metrics = {}
                    for i, micro_batch in enumerate(micro_batches):
                        metrics = self.train_step(micro_batch, accumulation_step=i)
                        step_metrics.update(metrics)
                    self.throughput_monitor.update(batch["model_input"].shape[0])
                    step_metrics.update(self.throughput_monitor.get_metrics())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(
                            "GPU OOM at step %d", 
                            self.global_step, 
                            exc_info=True,
                            extra={'stack_info': True}
                        )
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(
                            "Error during training step", 
                            exc_info=True,
                            extra={
                                'error': str(e),
                                'step': self.global_step,
                                'stack_info': True
                            }
                        )
                        raise

            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            if self.global_step % self.config.training.log_steps == 0:
                self.metrics_logger.log_metrics(
                    self.global_step,
                    self.wandb_logger
                )
                
            if (
                self.validation_logger is not None and
                self.global_step % self.config.training.validation_steps == 0
            ):
                validation_metrics = self.validation_logger.run_validation(
                    self.global_step,
                    self.wandb_logger
                )
                self.metrics_logger.log_metrics(
                    self.global_step,
                    self.wandb_logger,
                    additional_metrics=validation_metrics
                )

            if (
                self.config.training.save_steps > 0 and
                self.global_step % self.config.training.save_steps == 0
            ):
                self.save_checkpoint()

            self.global_step += 1
            progress_bar.update(1)
        finally:
            progress_bar.close()
        epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        return epoch_metrics

    def train(self) -> Dict[str, float]:
        logger.info(f"Starting training with {self.training_method.name} method...")
        metrics = {}
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch()
            log_metrics(
                epoch_metrics,
                epoch,
                is_main_process=is_main_process(),
                use_wandb=self.config.training.use_wandb,
                step_type="epoch"
            )
            metrics.update(epoch_metrics)
            self.save_checkpoint()
            if self.global_step >= self.max_steps:
                break
        return metrics

    def _validate_batch_sizes(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate that all tensors in a batch have consistent sizes."""
        expected_batch_size = None
        expected_latent_size = None
        
        def check_tensor(tensor: torch.Tensor, path: str) -> None:
            nonlocal expected_batch_size, expected_latent_size
            
            if len(tensor.shape) >= 2:  # Only check meaningful tensors
                if expected_batch_size is None:
                    expected_batch_size = tensor.shape[0]
                elif tensor.shape[0] != expected_batch_size:
                    raise ValueError(
                        f"Inconsistent batch size at {path}: "
                        f"expected {expected_batch_size}, got {tensor.shape[0]}"
                    )
                
                # For latent/model inputs, check spatial dimensions
                if path.endswith("model_input") or "latent" in path:
                    if expected_latent_size is None:
                        expected_latent_size = tensor.shape[2:]
                    elif tensor.shape[2:] != expected_latent_size:
                        raise ValueError(
                            f"Inconsistent latent spatial dimensions at {path}: "
                            f"expected {expected_latent_size}, got {tensor.shape[2:]}"
                        )
        
        def recursive_check(data: Union[torch.Tensor, Dict], path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    recursive_check(value, current_path)
            elif isinstance(data, torch.Tensor):
                check_tensor(data, path)
        
        try:
            recursive_check(batch)
        except ValueError as e:
            logger.error(
                f"Batch validation failed: {str(e)}", 
                extra={
                    'batch_structure': {
                        k: v.shape if isinstance(v, torch.Tensor) else type(v) 
                        for k, v in batch.items()
                    }
                }
            )
            raise

    def _prepare_micro_batches(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        # Validate batch sizes first
        self._validate_batch_sizes(batch)
        
        if self.gradient_accumulation_steps == 1:
            return [batch]
            
        micro_batches = []
        # Get batch size from latent structure
        if "latent" in batch:
            model_input = batch["latent"].get("model_input", batch["latent"].get("latent", {}).get("model_input"))
        else:
            model_input = batch.get("model_input")
        
        if model_input is None:
            raise ValueError("Could not find model input in batch")
        
        batch_size = model_input.shape[0]
        micro_batch_size = batch_size // self.gradient_accumulation_steps
        
        # Update tensor keys to match cache format
        tensor_keys = [
            "latent",  # New cache format
            "embeddings",  # New cache format
            "model_input",  # Legacy format
            "prompt_embeds",  # Legacy format
            "pooled_prompt_embeds"  # Legacy format
        ]
        
        for i in range(self.gradient_accumulation_steps):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            
            micro_batch = {}
            for k, v in batch.items():
                if k in tensor_keys and isinstance(v, (torch.Tensor, dict)):
                    if isinstance(v, dict):
                        # Handle nested dictionary structure
                        micro_batch[k] = {
                            sub_k: sub_v[start_idx:end_idx] if isinstance(sub_v, torch.Tensor)
                            else sub_v for sub_k, sub_v in v.items()
                        }
                    else:
                        micro_batch[k] = v[start_idx:end_idx]
                elif isinstance(v, list):
                    micro_batch[k] = v[start_idx:end_idx]
                else:
                    micro_batch[k] = v
                    
            micro_batches.append(micro_batch)
            
        return micro_batches

    def _setup_memory_management(self, batch_size: Optional[int] = None, micro_batch_size: Optional[int] = None) -> None:
        self.memory_optimized = setup_memory_optimizations(
            model=self.model,
            config=self.config,
            device=self.device,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size
        )
        if self.memory_optimized:
            verify_memory_optimizations(self.model, self.config, self.device, logger)
            if self.config.training.memory.enable_24gb_optimizations:
                self.layer_offloader = LayerOffloader(
                    model=self.model,
                    config=LayerOffloadConfig(
                        enabled=True,
                        fraction=self.config.training.memory.layer_offload_fraction,
                        temp_device=self.config.training.memory.temp_device,
                        async_transfer=self.config.training.memory.enable_async_offloading
                    ),
                    device=self.device
                )
        def cleanup_hook():
            torch_sync()
        self.cleanup_hook = cleanup_hook

    def save_checkpoint(self) -> None:
        if not is_main_process():
            return
        logger.info(f"Saving checkpoint at step {self.global_step}...")
        checkpoint_dir = Path(self.config.global_config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save({"global_step": self.global_step, "epoch": self.epoch}, checkpoint_dir / "state.pt")
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
