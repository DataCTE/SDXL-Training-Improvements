"""Base optimizer class for SDXL training."""
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Iterable, Dict, Any
import torch

# Sentinel value for required parameters
required = object()

class BaseOptimizer(ABC):
    """Base class for all SDXL optimizers."""
    
    def __init__(self, params: Iterable[torch.Tensor], defaults: Dict[str, Any]) -> None:
        """Initialize optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            defaults: Dict containing default values of optimization options
        """
        self.defaults = defaults
        self._param_groups = list(params)
        
        # Initialize state dict for optimizer
        self.state: Dict[torch.Tensor, Dict[str, Any]] = {}
        
        # Add parameters to param_groups
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument should be an iterable of Tensors or dicts")
            
        self.param_groups = []
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
            
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
            
        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a param group to the optimizer's param groups.
        
        Args:
            param_group: Dict containing parameters to add
        """
        assert isinstance(param_group, dict), "param group must be a dict"
        
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                          'the ordering of tensors in sets will be undefined. '
                          'Please use a list instead.')
        else:
            param_group['params'] = list(params)
            
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                              f"but one of the params is {type(param)}")
                
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(f"parameter group didn't specify a value of required optimization parameter {name}")
            else:
                param_group.setdefault(name, default)
                
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
            
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")
            
        self.param_groups.append(param_group)

    @abstractmethod
    def step(self, closure: Optional[callable] = None) -> None:
        """Performs a single optimization step."""
        pass

    @abstractmethod
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears the gradients of all optimized parameters."""
        pass

    @property
    @abstractmethod
    def param_groups(self) -> Iterator[dict]:
        """Returns an iterator over parameter groups."""
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """Returns the optimizer state as a dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the optimizer state."""
        pass 