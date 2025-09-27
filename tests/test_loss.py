import src as M
import torch
import pytest
from src.activations import *
from src.losses import *
import numpy as np


@pytest.mark.parametrize("pred, target", [
    ([0, 1, 2], [0, 1, 2]),                  # iguales
    ([1, 2, 3], [3, 2, 1]),                  # invertidos
    ([0.5, 1.5, 2.5], [1, 1, 1]),            # valores decimales
    (np.random.uniform(-5, 5, 5), np.random.uniform(-5, 5, 5)), # aleatorio
])
def test_mse(pred, target):
    mse = MSE()
    a = M.Tensor(pred, requires_grad=True)
    b = M.Tensor(target, requires_grad=False)

    loss = mse(a, b)
    ta = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(target, dtype=torch.float32)

    ref_loss = torch.nn.functional.mse_loss(ta, tt)
    loss.sum().backward()
    ref_loss.backward()

    assert np.allclose(loss.data, ref_loss.detach().numpy(), atol=1e-5)
    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("pred, target", [
    ([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], [0, 1]),  # Índices enteros
    ([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], [[1, 0, 0], [0, 1, 0]]),  # One-hot
    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [2, 1]),  # Índices enteros
    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0, 0, 1], [0, 1, 0]]),  # One-hot
])
def test_cross_entropy(pred, target):
    ce = CrossEntropy()

    # MiniTorch
    a = M.Tensor(pred, requires_grad=True)
    b = M.Tensor(target, requires_grad=False)
    loss = ce(a, b)

    # PyTorch
    ta = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
    if isinstance(target[0], list):  # One-hot
        tt = torch.tensor(target, dtype=torch.float32)
        ref_loss = -(tt * torch.nn.functional.log_softmax(ta, dim=1)).sum(dim=1).mean()
    else:  # Índices enteros
        tt = torch.tensor(target, dtype=torch.long)
        ref_loss = torch.nn.functional.cross_entropy(ta, tt)

    # Backward
    loss.backward()
    ref_loss.backward()

    # Verificaciones
    assert np.allclose(loss.data, ref_loss.detach().numpy(), atol=1e-5)
    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])

if __name__ == "__main__":
    pytest.main([__file__])