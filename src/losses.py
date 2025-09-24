from src.base import Function
from src.tensor import Tensor

class MSE(Function):
    def forward(self, prediction, target):
        """Calculates the Mean Squared Error between prediction and target tensors

        Args:
            prediction (Tensor): The predicted output tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The resulting tensor after calculating the MSE.
        """
        target = target if isinstance(target, Tensor) else Tensor(target)

        return ((prediction - target) ** 2).mean()