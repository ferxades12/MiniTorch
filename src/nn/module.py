
class Module:
    def __init__(self):
        self.submodules = []
        self._parameters = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self):
        raise  NotImplementedError

    def parameters(self):
        params = list(self._parameters)

        for module in self.submodules:
            params.extend(module.parameters())

        return params

    