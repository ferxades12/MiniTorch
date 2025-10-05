from src.nn.activations import ReLU, Sigmoid, Tanh, Softmax
from src.nn.losses import MSELoss, CrossEntropyLoss
from src.nn.regularizations import L1_Reg, L2_Reg

def relu(tensor):
    """Applies the ReLU activation function element-wise.

    Args:
        tensor (array_like): The input tensor to apply ReLU to.
    Returns:
        Tensor: The resulting tensor after applying the ReLU function.
    """
    return ReLU()(tensor)

def sigmoid(tensor):
    """Applies the sigmoid activation function element-wise.

    Args:
        tensor (array_like): The input tensor to apply sigmoid to.  
    Returns:
        Tensor: The resulting tensor after applying the sigmoid function.
    """
    return Sigmoid()(tensor)

def tanh(tensor):
    """Applies the hyperbolic tangent activation function element-wise.

    Args:
        tensor (array_like): The input tensor to apply tanh to.
    Returns:
        Tensor: The resulting tensor after applying the tanh function.
    """
    return Tanh()(tensor)

def softmax(tensor):
    """Applies the softmax activation function to a tensor. Currently only supports 1D tensors.

    Args:
        tensor (array_like): The input tensor to apply softmax to.
    Raises:
        ValueError: If the input tensor is not 1D.
    Returns:
        Tensor: The resulting tensor after applying the softmax function.
    """
    return Softmax()(tensor)

def mse(predictions, targets):
    """Computes the Mean Squared Error (MSE) loss between predictions and targets.

    Args:
        predictions (array_like): The predicted values.
        targets (array_like): The true target values.
    Returns:
        Tensor: The computed MSE loss.
    """
    return MSELoss()(predictions, targets)

def cross_entropy(predictions, targets):
    """Computes the Cross-Entropy loss between predictions and targets.

    Args:
        predictions (array_like): The predicted values (logits).
        targets (array_like): The true target values (class indices).
    Returns:
        Tensor: The computed Cross-Entropy loss.
    """
    return CrossEntropyLoss()(predictions, targets)


def l1(loss, model, l:float = 1e-5):
    """Applies L1 regularization to the given loss based on the model's weights.

    Args:
        loss (Tensor): The original loss tensor.
        model (nn.Module): The model being regularized.
        l (float, optional): Weight decay factor. Defaults to 1e-5.
    Returns:
        Tensor: The loss tensor with L1 regularization added.
    """
    return L1_Reg(l)(loss, model)

def l2(loss, model, l:float = 1e-5):
    """Applies L2 regularization to the given loss based on the model's weights.

    Args:
        loss (Tensor): The original loss tensor.
        model (nn.Module): The model being regularized.
        l (float, optional): Weight decay factor. Defaults to 1e-5.
    Returns:
        Tensor: The loss tensor with L2 regularization added.
    """
    return L2_Reg(l)(loss, model)