"""UNet model wrapper with optimized memory usage."""
import logging
import torch
from diffusers import UNet2DConditionModel
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class UNetWrapper(torch.nn.Module):
    """Wrapper around UNet2DConditionModel with memory optimizations."""
    
    def __init__(
        self,
        pretrained_model_name: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **model_kwargs: Any
    ):
        """Initialize UNet wrapper.
        
        Args:
            pretrained_model_name: HuggingFace model name
            device: Target device
            dtype: Model dtype
            **model_kwargs: Additional kwargs for model
        """
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # Load model
        try:
            self.model = UNet2DConditionModel.from_pretrained(
                pretrained_model_name,
                subfolder="unet",
                torch_dtype=self.dtype,
                **model_kwargs
            )
            self.model.to(device=self.device)
            
        except Exception as e:
            logger.error(f"Failed to load UNet model: {str(e)}")
            raise
            
        # Configure memory format
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with automatic device/dtype handling."""
        # Move inputs to model device/dtype
        args = [
            arg.to(device=self.device, dtype=self.dtype) 
            if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device=self.device, dtype=self.dtype)
                
        return self.model(*args, **kwargs)
        
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict."""
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load model state dict."""
        self.model.load_state_dict(state_dict)
        
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
