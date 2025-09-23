"""
Standard activation functions for MiniTorch.
"""

import src as M
from src.tensor import Tensor
from src.operations import Sigmoid, Softmax1D

def ReLU(tensor):
    """Applies the ReLU function element-wise using the maximum operation

    Args:
        tensor (any type convertible to Tensor): The input tensor to apply ReLU to.

    Returns:
        Tensor: The resulting tensor after applying the ReLU function.
    """
    return M.maximum(tensor, 0)

def sigmoid(tensor):
    """Applies the sigmoid function element-wise

    Args:
        tensor (any type convertible to Tensor): The input tensor to apply sigmoid to.

    Returns:
        Tensor: The resulting tensor after applying the sigmoid function.
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return tensor._apply_unary_op(Sigmoid)

def tanh(tensor):
    """Applies the hyperbolic tangent function element-wise using the sigmoid function

    Args:
        tensor (any type convertible to Tensor): The input tensor to apply tanh to.

    Returns:
        Tensor: The resulting tensor after applying the tanh function.
    """
    return 2 * sigmoid(2 * tensor) - 1

def softmax(tensor):
    """Applies the softmax function to a tensor. Currently only supports 1D tensors

    Args:
        tensor (any type convertible to Tensor): The input tensor to apply softmax to.
    Raises:
        ValueError: If the input tensor is not 1D.

    Returns:
        Tensor: The resulting tensor after applying the softmax function.
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    
    if tensor.numdims() == 1:
        return tensor._apply_unary_op(Softmax1D)
    else:
        raise ValueError("Softmax not implemented for more than 1 dim")