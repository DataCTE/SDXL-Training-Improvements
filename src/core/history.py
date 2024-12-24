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
