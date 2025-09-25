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



if __name__ == "__main__":
    pytest.main([__file__])