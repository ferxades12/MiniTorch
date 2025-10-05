from src.tensor import Tensor

class Module:
    def __init__(self):
        """Base class for all neural networks and layers.
        """
        self.submodules = {}
        self.params = []
        self.training: bool = True

    def __call__(self, *args):
        return self.forward(*args)
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if isinstance(value, Module):
            self.submodules[name] = value

    def forward(self):
        raise  NotImplementedError

    def parameters(self) -> list:
        """Return the list of all parameters in the module and its submodules.
        """
        params = self.params

        for module in self.submodules.values():
            params.extend(module.parameters())

        return params

    def get_weights(self) -> list[Tensor]:
        weights = []

        for module in self.submodules.values():
            weights.extend(module.weight)

        return weights
    
    def eval(self) -> None:
        self.training = False

        for module in self.submodules.values():
            module.training = False
    
    def train(self) -> None:
        self.training = True

        for module in self.submodules.values():
            module.training = True

    