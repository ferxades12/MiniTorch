import src as M
import torch
import pytest
from src.nn.activations import *
from src.nn.losses import *
import numpy as np


@pytest.mark.parametrize("base, exp", [
    ([[1, 2, 3]], [[0.5, 1.5, -1]]), 
    ([[1e-7, 1, 10]], [[-2, 0, 2]]),
    ([[2, 3, 4]], 2),
    (np.random.uniform(0.1, 5, (3, 3)), np.random.uniform(-2, 2, (3, 3))),
])
def test_pow(base, exp):
    A = M.Tensor(base, requires_grad=True)
    B = M.Tensor(exp, requires_grad=True)

    C = A ** B
    ta = torch.tensor(base, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(exp, dtype=torch.float32, requires_grad=True)
    tc = ta ** tb

    C.sum().backward()
    tc.sum().backward()

    assert np.allclose(C.data, tc.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and A.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    if B.numdims() > 0:
        assert tb.grad is not None and B.grad is not None and np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("a, b", [
    ([[1, 2, 3]], [[4, 5, 6]]),                # vectores fila
    ([[1e-7, 1, 10]], [[-2, 0, 2]]),           # valores pequeños, cero y negativos
    ([[2, 3, 4]], 2),                          # tensor por escalar
    (np.random.uniform(-5, 5, (3, 3)), np.random.uniform(-2, 2, (3, 3))), # aleatorio
])
def test_mul(a, b):
    A = M.Tensor(a, requires_grad=True)
    B = M.Tensor(b, requires_grad=True) if isinstance(b, (list, np.ndarray)) else b

    C = A * B
    ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float32, requires_grad=True) if isinstance(b, (list, np.ndarray)) else b
    tc = ta * tb

    C.sum().backward()
    tc.sum().backward()

    assert np.allclose(C.data, tc.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and A.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    if isinstance(B, M.Tensor):
        assert tb.grad is not None and B.grad is not None and np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[1, 2, 3], [4, 5, 6]],                  # matriz 2x3
    [[1, 2], [3, 4], [5, 6]],                # matriz 3x2
    [[1, 2, 3, 4]],                          # matriz 1x4
    np.random.uniform(-5, 5, (4, 2)),        # matriz aleatoria 4x2
    [[7]],                                   # matriz 1x1
])
def test_transpose(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = A.T()
    tb = ta.t()

    B.sum().backward()
    tb.sum().backward()

    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and A.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("a, b", [
    ([1, 2, 3], [4, 5, 6]),                      # vectores 1D
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),        # matrices 2D
    ([[1, 2, 3]], [[4], [5], [6]]),              # fila por columna
    (np.random.uniform(-5, 5, (2, 3)), np.random.uniform(-2, 2, (3, 4))), # aleatorio compatible
])
def test_dot(a, b):
    A = M.Tensor(a, requires_grad=True)
    B = M.Tensor(b, requires_grad=True)

    C = A.dot(B)
    ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float32, requires_grad=True)
    tc = ta @ tb

    C.sum().backward()
    tc.sum().backward()

    assert np.allclose(C.data, tc.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and A.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert tb.grad is not None and B.grad is not None and np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [1, 2, 3],                  # Valores positivos simples
    [0.1, 1, 10],               # Valores pequeños y grandes
    [1e-7, 1, 1e7],             # Valores extremos
    np.random.uniform(0.1, 5, 10),  # Valores aleatorios positivos
])
def test_log(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    # Aplica log en MiniTorch y PyTorch
    B = A.log()
    tb = torch.log(ta)

    # Calcula el backward
    B.sum().backward()
    tb.sum().backward()

    # Verifica que los resultados sean iguales
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and A.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])

