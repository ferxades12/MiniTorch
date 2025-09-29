"""
Standard activation functions for MiniTorch.
"""

import src as M
from src.tensor import Tensor
from src.operations import SigmoidOp, SoftmaxOp
from src.base import Function


class ReLU(Function):
    def forward(self, tensor) -> Tensor:
        """Applies the ReLU function element-wise using the maximum operation

        Args:
            tensor (any type convertible to Tensor): The input tensor to apply ReLU to.

        Returns:
            Tensor: The resulting tensor after applying the ReLU function.
        """
        return M.maximum(tensor, 0)


class Sigmoid(Function):
    def forward(self, tensor) -> Tensor:
        """Applies the sigmoid function element-wise

        Args:
            tensor (any type convertible to Tensor): The input tensor to apply sigmoid to.

        Returns:
            Tensor: The resulting tensor after applying the sigmoid function.
        """
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        return tensor._apply_unary_op(SigmoidOp)


class Tanh(Function):
    def forward(self, tensor) -> Tensor:
        """Applies the hyperbolic tangent function element-wise using the sigmoid function

        Args:
            tensor (any type convertible to Tensor): The input tensor to apply tanh to.

        Returns:
            Tensor: The resulting tensor after applying the tanh function.
        """
        return 2 * Sigmoid()(2 * tensor) - 1


class Softmax(Function):
    def forward(self, tensor) -> Tensor:
        """Applies the softmax function to a tensor. Currently only supports 1D tensors

        Args:
            tensor (any type convertible to Tensor): The input tensor to apply softmax to.
        Raises:
            ValueError: If the input tensor is not 1D.

        Returns:
            Tensor: The resulting tensor after applying the softmax function.
        """
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        
        return tensor._apply_unary_op(SoftmaxOp)
