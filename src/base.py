
class Function():
    def __call__(self, tensor):
        return self.forward(tensor)
    
    def forward(self):
        raise NotImplementedError