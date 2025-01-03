"""Base optimizer class for SDXL training."""
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Iterable, Dict, Any, List
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
        self.state: Dict[torch.Tensor, Dict[str, Any]] = {}
        self._param_groups: List[Dict[str, Any]] = []
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
            
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
            
        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a param group to the optimizer's param groups."""
        assert isinstance(param_group, dict), "param group must be a dict"
        
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections')
        else:
            param_group['params'] = list(params)
            
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors")
                
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(f"parameter group didn't specify required parameter {name}")
            else:
                param_group.setdefault(name, default)
                
        param_set = set()
        for group in self._param_groups:
            param_set.update(set(group['params']))
            
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")
            
        self._param_groups.append(param_group)

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Returns the parameter groups."""
        return self._param_groups

    @abstractmethod
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns:
            Optional loss value if closure is provided.
        """
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer state as a dict."""
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state."""
        self.state = state_dict['state']
        self._param_groups = state_dict['param_groups'] 