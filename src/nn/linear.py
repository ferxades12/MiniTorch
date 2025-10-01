from src.nn.module import Module
import numpy as np
from src.tensor import Tensor

class Linear(Module):
    def __init__(self, in_features:int, out_features:int, bias:bool=True):
        """_summary_

        Args:
            in_features (int): Number of input features (weights)
            out_features (int): Number of output features (neurons)
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__()
        k = np.sqrt(1 / in_features)
        size = (in_features, out_features)
        self.weight = Tensor(np.random.uniform(-k, k, size), requires_grad=True)

        self.parameters.append(self.weight)

        if bias:
            self.bias = Tensor(np.random.uniform(-k, k, out_features), requires_grad=True)
            self.parameters.append(self.bias)
       
        