from src.nn.module import Module
import numpy as np
from src.tensor import Tensor
from src.ops import DropoutOp

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
        self.weight:Tensor = Tensor(np.random.uniform(-k, k, size), requires_grad=True)

        self.params.append(self.weight)

        if bias:
            self.bias:Tensor = Tensor(np.random.uniform(-k, k, out_features), requires_grad=True)
            self.params.append(self.bias)
        else:
            self.bias = None
       
    def forward(self, x:Tensor) -> Tensor:
        """Applies a linear transformation to the incoming data: y = xW + b
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features)

        Raises:
            ValueError: If the input tensor's last dimension does not match in_features.
        """

        if x.shape()[-1] != self.weight.shape()[0]:
            raise ValueError("Variable dimensions are not correct")
        
        x = x if isinstance(x, Tensor) else Tensor(x)
        bias = self.bias if self.bias is not None else 0

        return x.dot(self.weight) + bias
    
class Dropout(Module):
    def __init__(self, p:float = 0.5):
        """Dropout regularization layer.

        Args:
            p (float, optional): Probability of dropping out a neuron. Defaults to 0.5
        """
        super().__init__()
        self.p = p
    
    def forward(self, x:Tensor) -> Tensor:
        """Applies dropout to the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with dropout applied.
        """
        return x._apply_unary_op(DropoutOp, self.p, self.training)


class Sequential(Module):
    def __init__(self, *args : Module):
        super().__init__()
        

        for module in args:
            if not isinstance(module, Module):
                raise ValueError("All arguments must be instances of Module")

            self.submodules.append(module)
       
    def forward(self, x:Tensor) -> Tensor:
        for module in self.submodules:
            x = module(x)
        
        return x
    
    def add_module(self, module:Module) -> None:
        if not isinstance(module, Module):
            raise ValueError("Argument must be an instance of Module")
        
        self.submodules.append(module)