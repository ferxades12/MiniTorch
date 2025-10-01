
class Module:
    def __init__(self):
        self.submodules = []
        self.params = []

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self):
        raise  NotImplementedError

    def parameters(self):
        params = self.params

        for module in self.submodules:
            params.extend(module.parameters())

        return params 

    