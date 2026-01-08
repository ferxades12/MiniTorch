
class Function():
    """Base class for all operations in the autograd system. 
    Each operation should inherit from this class and implement the forward and backward methods.
    """
    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, *args):
        raise NotImplementedError