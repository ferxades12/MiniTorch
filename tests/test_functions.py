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

def test_pow():
    a = M.Tensor([[1, 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    b = M.Tensor([[2, 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)

    c = a**b
    d = c.T()
    e = d**a

    x = e.sum()
    x.backward()

    ta = torch.tensor([[1., 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    tb = torch.tensor([[2., 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)
    tc = ta**tb
    td = tc.T

    te = td**ta
    tx = te.sum()
    tx.backward()

    r, c =  ta.shape
    for i in range(r):
        for j in range(c):
            assert a.grad[i, j] == ta.grad[i, j]
            assert b.grad[i, j] == tb.grad[i, j]

def test_ReLU():
    A = M.Tensor([[-1, 0], [1, -1]], requires_grad=True)
    ta = torch.tensor([[-1., 0], [1, -1]], requires_grad=True)

    B = act.ReLU(A)
    tb = torch.nn.functional.relu(ta)

    B.sum().backward()
    tb.sum().backward()

    r, c =  tb.shape
    for i in range(r):
        for j in range(c):
            assert A.grad[i, j] == ta.grad[i, j]
            assert B.data[i, j] == tb[i, j]

def test_sigmoid():
    A = M.Tensor([[-1, 0], [1, -1]], requires_grad=True)
    ta = torch.tensor([[-1., 0], [1, -1]], requires_grad=True)

    B = act.sigmoid(A)
    tb = torch.sigmoid(ta)

    B.sum().backward()
    tb.sum().backward()

    print(A.grad)
    print(ta.grad)

    r, c =  tb.shape
    for i in range(r):
        for j in range(c):
            assert A.grad[i, j] - ta.grad[i, j] < 1e-4
            assert B.data[i, j] - tb[i, j]< 1e-4

    

if __name__ == "__main__":
    pytest.main([__file__])

