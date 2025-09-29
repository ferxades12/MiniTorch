import src as M
import torch
import pytest
from src.nn.activations import *
from src.nn.losses import *
from src.nn.optimizers import *
import numpy as np

@pytest.mark.parametrize("lr", [0.01, 0.1])
@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("dampening", [0.0, 0.5])
@pytest.mark.parametrize("maximize", [False, True])
def test_sgd_parametrized(lr, momentum, dampening, maximize):
    np.random.seed(42)
    # Datos de entrada
    x_np = np.random.randn(4, 3)
    w1_np = np.random.randn(3, 2)
    b1_np = np.random.randn(2)

    # MiniTorch
    x = M.Tensor(x_np, requires_grad=True)
    w1 = M.Tensor(w1_np, requires_grad=True)
    b1 = M.Tensor(b1_np, requires_grad=True)

    # PyTorch
    tx = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    tw1 = torch.tensor(w1_np, dtype=torch.float32, requires_grad=True)
    tb1 = torch.tensor(b1_np, dtype=torch.float32, requires_grad=True)

    # Forward
    out = x.dot(w1) + b1
    loss = out.sum()
    tout = tx @ tw1 + tb1
    tloss = tout.sum()

    # Backward
    loss.backward()
    tloss.backward()

    # SGD MiniTorch
    optimizer = SGD([x, w1, b1], lr=lr, momentum=momentum, dampening=dampening, maximize=maximize)
    optimizer.step()

    # SGD PyTorch
    torch_optimizer = torch.optim.SGD([tx, tw1, tb1], lr=lr, momentum=momentum, dampening=dampening, maximize=maximize)
    torch_optimizer.step()

    print("Después paso - MiniTorch:", x.data)
    print("Después paso - PyTorch:", tx.detach().numpy())
    # Comprobaciones
    assert np.allclose(x.data, tx.detach().numpy(), atol=1e-5)
    assert np.allclose(w1.data, tw1.detach().numpy(), atol=1e-5)
    assert np.allclose(b1.data, tb1.detach().numpy(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])