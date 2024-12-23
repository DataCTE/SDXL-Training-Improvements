"""Layer offloading utilities for memory optimization with extreme speedups."""
import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

# Force maximal speed
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

from .tensor import (
    tensors_to_device_,
    torch_sync,
    create_stream_context,
    tensors_record_stream,
    device_equals
)

logger = logging.getLogger(__name__)

@dataclass
class LayerOffloadConfig:
    enabled: bool = False
    fraction: float = 0.0
    temp_device: str = "cpu"
    async_transfer: bool = True

class LayerOffloader:
    def __init__(
        self,
        model: nn.Module,
        config: LayerOffloadConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.temp_device = torch.device(config.temp_device)
        self.layer_map: Dict[str, torch.device] = {}
        self.streams: Dict[str, Optional[torch.cuda.Stream]] = {}
        if config.enabled:
            self._setup_offloading()

    def _setup_offloading(self):
        if not self.config.enabled:
            return
        if self.config.async_transfer and torch.cuda.is_available():
            self.streams["transfer"] = torch.cuda.Stream()
        else:
            self.streams["transfer"] = None
        total_params = sum(p.numel() for p in self.model.parameters())
        current_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params = sum(p.numel() for p in module.parameters())
                current_params += params
                if current_params / total_params > (1 - self.config.fraction):
                    self.layer_map[name] = self.temp_device
                else:
                    self.layer_map[name] = self.device

    def offload_layer(self, name: str):
        if not self.config.enabled or name not in self.layer_map:
            return
        module = dict(self.model.named_modules())[name]
        target_device = self.layer_map[name]
        if not device_equals(next(module.parameters()).device, target_device):
            if self.streams["transfer"] is not None:
                with create_stream_context(self.streams["transfer"]):
                    tensors_to_device_(module.state_dict(), target_device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.streams["transfer"])
            else:
                tensors_to_device_(module.state_dict(), target_device)
            torch_sync()

    def prefetch_layer(self, name: str):
        if not self.config.enabled or name not in self.layer_map:
            return
        module = dict(self.model.named_modules())[name]
        if not device_equals(next(module.parameters()).device, self.device):
            if self.streams["transfer"] is not None:
                with create_stream_context(self.streams["transfer"]):
                    tensors_to_device_(module.state_dict(), self.device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.streams["transfer"])
            else:
                tensors_to_device_(module.state_dict(), self.device)
            torch_sync()
