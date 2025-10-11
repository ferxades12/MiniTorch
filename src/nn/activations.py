"""
Standard activation functions for MiniTorch.
"""

import src as M
from src.tensor import Tensor
import src.nn as nn
from src.operations.operations import SigmoidOp, SoftmaxOp


class ReLU(nn.Module):
    def forward(self, tensor) -> Tensor:
        """Applies the ReLU function element-wise using the maximum operation

        Args:
            tensor (array_like): The input tensor to apply ReLU to.

        Returns:
            Tensor: The resulting tensor after applying the ReLU function.
        """
        return M.maximum(tensor, 0)


class Sigmoid(nn.Module):
    def forward(self, tensor) -> Tensor:
        """Applies the sigmoid function element-wise

        Args:
            tensor (array_like): The input tensor to apply sigmoid to.

        Returns:
            Tensor: The resulting tensor after applying the sigmoid function.
        """
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        return tensor._apply_unary_op(SigmoidOp)


class Tanh(nn.Module):
    def forward(self, tensor) -> Tensor:
        """Applies the hyperbolic tangent function element-wise using the sigmoid function

        Args:
            tensor (array_like): The input tensor to apply tanh to.

        Returns:
            Tensor: The resulting tensor after applying the tanh function.
        """
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)

        return 2 * Sigmoid()(2 * tensor) - 1


class Softmax(nn.Module):
    def forward(self, tensor) -> Tensor:
        """Applies the softmax function to a tensor. Currently only supports 1D tensors

        Args:
            tensor (array_like): The input tensor to apply softmax to.
        Raises:
            ValueError: If the input tensor is not 1D.

        Returns:
            Tensor: The resulting tensor after applying the softmax function.
        """
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        
        return tensor._apply_unary_op(SoftmaxOp)
