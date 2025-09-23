import src as M
import torch
import pytest
import src.activations as act
import numpy as np



def test_ops():
    a = M.Tensor([[1, 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    b = M.Tensor([[2, 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)

    c = a.dot(b)
    d = c.T()
    e = d.dot(a)

    x = e.sum()
    x.backward()

    ta = torch.tensor([[1., 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    tb = torch.tensor([[2., 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)
    tc = torch.tensordot(ta, tb, dims = 1)
    td = tc.T

    te = torch.tensordot(td, ta, dims = 1)
    tx = te.sum()
    tx.backward()

    r, c =  ta.shape
    for i in range(r):
        for j in range(c):
            assert a.grad[i, j] == ta.grad[i, j]
            assert b.grad[i, j] == tb.grad[i, j]

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
    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    if B.numdims() > 0:
        assert np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

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
    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    if isinstance(B, M.Tensor):
        assert np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

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
    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

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
    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],           # negativos, ceros y positivos
    [[0, 0, 0]],                  # solo ceros
    [[-1000, 0, 1000]],           # extremos
    [[-1e-7, 0, 1e-7]],           # valores muy pequeños
    np.random.uniform(-5, 5, (3, 3)), # aleatorio
])
def test_relu(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = act.ReLU(A)
    tb = torch.nn.functional.relu(ta)

    B.sum().backward()
    tb.sum().backward()

    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],
    [[-10, -1, 0, 1, 10]],
    [[-100, -0.5, 0, 0.5, 100]],
    [[-2, -0.1, 0], [0.1, 2, 5]],
    np.random.uniform(-5, 5, (3, 3)),
])
def test_sigmoid(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = act.sigmoid(A)
    tb = torch.sigmoid(ta)

    B.sum().backward()
    tb.sum().backward()

    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],
    [[-10, -1, 0, 1, 10]],
    [[-100, -0.5, 0, 0.5, 100]],
    [[-2, -0.1, 0], [0.1, 2, 5]],
    np.random.uniform(-5, 5, (3, 3)),
])
def test_tanh(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = act.tanh(A)
    tb = torch.tanh(ta)

    B.sum().backward()
    tb.sum().backward()

    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)


@pytest.mark.parametrize("arr", [
    [0, 1, 2, 3],                # valores crecientes
    [-1, 0, 1],                  # negativos y positivos
    [1000, 1001, 1002],          # valores grandes (test de estabilidad numérica)
    [-1000, 0, 1000],            # extremos
    [0, 0, 0],                   # todos iguales
    np.random.uniform(-5, 5, 5), # aleatorio
])
def test_softmax_1d(arr):
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = act.softmax(A)
    tb = torch.nn.functional.softmax(ta, dim=0)

    B.sum().backward()
    tb.sum().backward()

    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])

