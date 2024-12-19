"""Layer offloading utilities for memory optimization."""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass 
class LayerOffloadConfig:
    """Configuration for layer offloading."""
    enabled: bool = False
    fraction: float = 0.0
    temp_device: str = "cpu"
    async_transfer: bool = True
    
class LayerOffloader:
    """Handles offloading model layers to CPU/disk."""
    
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
        """Initialize offloading configuration."""
        if not self.config.enabled:
            return
            
        # Setup CUDA streams for async transfer
        if self.config.async_transfer and torch.cuda.is_available():
            self.streams["transfer"] = torch.cuda.Stream()
        else:
            self.streams["transfer"] = None
            
        # Map layers to devices
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
        """Offload a layer to temporary device."""
        if not self.config.enabled or name not in self.layer_map:
            return
            
        module = dict(self.model.named_modules())[name]
        target_device = self.layer_map[name]
        
        if next(module.parameters()).device != target_device:
            if self.streams["transfer"] is not None:
                with torch.cuda.stream(self.streams["transfer"]):
                    module.to(target_device)
                torch.cuda.current_stream().wait_stream(self.streams["transfer"])
            else:
                module.to(target_device)
                
    def prefetch_layer(self, name: str):
        """Prefetch a layer back to main device."""
        if not self.config.enabled or name not in self.layer_map:
            return
            
        module = dict(self.model.named_modules())[name]
        if next(module.parameters()).device != self.device:
            if self.streams["transfer"] is not None:
                with torch.cuda.stream(self.streams["transfer"]):
                    module.to(self.device)
                torch.cuda.current_stream().wait_stream(self.streams["transfer"])
            else:
                module.to(self.device)
