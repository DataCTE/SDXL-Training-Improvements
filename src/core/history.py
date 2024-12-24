import torch

def log_parameters_hook(module, inputs, outputs):
    # Access the 'history_instance' attribute of the module
    module.history_instance.log_parameters(module)

class TorchHistory:
    def __init__(self, module):
        self.module = module
        self.hook_handle = None

    def log_parameters(self, module):
        # Example implementation of logging parameters
        print(f"Logging parameters for module: {module}")

    def add_log_parameters_hook(self):
        # Associate the history instance with the module
        self.module.history_instance = self

        # Register the hook using the top-level function
        self.hook_handle = self.module.register_forward_hook(log_parameters_hook)

    def remove_log_parameters_hook(self):
        # Remove the hook when it's no longer needed
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __getstate__(self):
        """Return the state of the instance for pickling, excluding unpicklable attributes."""
        state = self.__dict__.copy()
        # Remove the hook_handle and module references, as they are not picklable
        state.pop('hook_handle', None)
        state.pop('module', None)
        return state

    def __setstate__(self, state):
        """Restore the state of the instance after unpickling."""
        self.__dict__.update(state)
        # Re-initialize attributes that were not pickled
        self.hook_handle = None
        self.module = None  # The module will need to be set manually after unpickling
