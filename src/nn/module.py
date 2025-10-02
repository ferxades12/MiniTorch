
class Module:
    def __init__(self):
        """Base class for all neural networks and layers.
        """
        self.submodules = []
        self.params = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self):
        raise  NotImplementedError

    def parameters(self):
        """Return the list of all parameters in the module and its submodules.
        """
        params = self.params

        for module in self.submodules:
            params.extend(module.parameters())

        return params 

    