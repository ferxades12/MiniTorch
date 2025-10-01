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

        self.params.append(self.weight)

        if bias:
            self.bias = Tensor(np.random.uniform(-k, k, out_features), requires_grad=True)
            self.params.append(self.bias)
        else:
            self.bias = None
       
    def forward(self, x:Tensor) -> Tensor:
        if x.shape()[-1] != self.weight.shape()[0]:
            raise ValueError("Variable dimensions are not correct")
        
        x = x if isinstance(x, Tensor) else Tensor(x)
        bias = self.bias if self.bias is not None else 0

        return x.dot(self.weight) + bias