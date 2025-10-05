import numpy as np
import src.nn as nn
from src.operations import DropoutOp
from src.tensor import Tensor


class Dropout(nn.Module):
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

class L1_Reg():
    def __init__(self, l:float = 1e-5):
        """L1 regularization layer.

        Args:
            l (float, optional): Weight decay factor. Defaults to 1e-5.
        """
        self.l = l

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, loss:Tensor, model) -> Tensor:
        """Computes the L1 loss.

        Args:
            loss (Tensor): The original loss tensor.
            model (nn.Module): The model being regularized.

        Returns:
            Tensor: The loss tensor with L1 regularization added.
        """
        l1_loss = loss
        for weight in model.get_weights():
            l1_loss += self.l * weight.abs().sum()
        
        return l1_loss
    
class L2_Reg():
    def __init__(self, l:float = 1e-5):
        """L2 regularization layer.
        Args:
            l (float, optional): Weight decay factor. Defaults to 1e-5.
        """
        self.l = l

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, loss:Tensor, model) -> Tensor:
        """Computes the L2 loss.
        Args:
            loss (Tensor): The original loss tensor.
            model (nn.Module): The model being regularized.
        Returns:
            Tensor: The loss tensor with L2 regularization added.
        """
        l2_loss = loss
        for weight in model.get_weights():
            l2_loss += self.l * (weight**2).sum()

        return l2_loss