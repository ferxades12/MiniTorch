from src.base import Function
from src.tensor import Tensor
from src.activations import Softmax
import numpy as np

class MSE(Function):
    def forward(self, prediction, target):
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
    def forward(self, prediction, target):
        """
        Calculates the Cross-Entropy loss between prediction and target tensors.

        If target is a vector Ex. [2, 1, 0]
            CE = - Σ log(Softmax(prediction)[i, target[i]])
            This selects the correct class based on the index in target for each sample
        
        If target is a one-hot tensor Ex. [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            CE = - Σ (target * log(Softmax(prediction)))
            The bit-wise multiplication implicitly selects the correct class for each sample
        
        Args:
            prediction (Tensor): The predicted logits tensor.
            target (Tensor or list): The ground truth target tensor (one-hot or class indices).

        Returns:
            Tensor: The resulting scalar tensor after calculating the Cross-Entropy loss.
        """
        target = target if isinstance(target, Tensor) else Tensor(target)
        s = Softmax()(prediction).log()

        if target.shape() == s.shape():
            # Target is one-hot
            return -1 * (target * s).sum(axis=1)
            

        else:
            # Target is a vector
            # np.arange(len(target.data)), target.data.astype(int)] = [i, target[i]] with i = 1,...,len(target.data)
            return -1 * s[np.arange(len(target.data)), target.data.astype(int)]