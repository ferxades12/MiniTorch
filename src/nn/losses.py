from src.base import Function
from src.tensor import Tensor
from src.operations import CrossEntropyOp

class MSELoss(Function):
    def forward(self, prediction : Tensor, target) -> Tensor:
        """Calculates the Mean Squared Error between prediction and target tensors

        Args:
            prediction (Tensor): The predicted logits tensor.
            target (Tensor): The ground truth target tensor.

        Returns:
            Tensor: The resulting tensor after calculating the MSE.
        """
        target = target if isinstance(target, Tensor) else Tensor(target)

        return ((prediction - target) ** 2).mean()

class CrossEntropyLoss(Function):
    def forward(self, tensor : Tensor, target) -> Tensor:
        """Calculates the Cross-Entropy loss between prediction and target tensors
        
        Args:
            tensor (Tensor): The predicted logits tensor.
            target (TensorLike): The target tensor, either as class indices or one-hot encoded.
        Returns:
            Tensor: The resulting tensor after calculating the Cross-Entropy loss.
        """
        return tensor._apply_binary_op(CrossEntropyOp, target)