
class Function():
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError