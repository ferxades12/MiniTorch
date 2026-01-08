from src.tensor import Tensor


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
        l1_penalty = sum(p.abs().sum() for p in model.get_weights() if p.numdims() > 1) # No se aplica a bias

        return loss + self.l * l1_penalty
    
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
        l2_penalty = sum((p**2).sum() for p in model.get_weights() if p.numdims() > 1) # No se aplica a bias

        return loss + self.l * l2_penalty