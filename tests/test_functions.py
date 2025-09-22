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

    assert_allclose(C.data, tc)
    assert_allclose(A.grad, ta.grad)
    assert_allclose(B.grad, tb.grad)

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

    assert_allclose(A.grad, ta.grad)
    assert_allclose(B.data, tb)

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

    assert_allclose(A.grad, ta.grad)
    assert_allclose(B.data, tb)

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

    assert_allclose(A.grad, ta.grad)
    assert_allclose(B.data, tb)


def assert_allclose(A, B):
    r, c =  A.shape
    for i in range(r):
        for j in range(c):
            assert abs((A[i, j] / B[i, j]) - 1) < 1e-3 or abs(A[i, j] - B[i, j]) < 1e-3

if __name__ == "__main__":
    pytest.main([__file__])

