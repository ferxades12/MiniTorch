from src.base import Function
from src.tensor import Tensor
from src.activations import Softmax
from src.operations import CrossEntropyOp
import numpy as np

class MSE(Function):
    def forward(self, prediction : Tensor, target):
        """Calculates the Mean Squared Error between prediction and target tensors

        Args:
            prediction (Tensor): The predicted logits tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The resulting tensor after calculating the MSE.
        """
        target = target if isinstance(target, Tensor) else Tensor(target)

        return ((prediction - target) ** 2).mean()

class CrossEntropy(Function):
    def forward(self, tensor : Tensor, target):
        target = target if isinstance(target, Tensor) else Tensor(target)

        return tensor._apply_binary_op(CrossEntropyOp, target)