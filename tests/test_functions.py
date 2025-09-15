import tensor as t
import torch
import pytest

def test():
    a = t.Tensor([[1, 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    b = t.Tensor([[2, 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)

    c = a.dot(b)
    d = c * 3
    e = d.dot(a)

    x = e.sum()
    x.backward()

    ta = torch.tensor([[1., 2, 3], [3, 2, 1], [3, 1, 2]], requires_grad=True)
    tb = torch.tensor([[2., 1, 1], [4, 2, 2], [1, 5, 6]], requires_grad=True)
    tc = torch.tensordot(ta, tb, dims = 1)
    td = tc * 3

    te = torch.tensordot(td, ta, dims = 1)
    x = te.sum()
    x.backward()

    r, c =  ta.shape
    for i in range(r):
        for j in range(c):
            assert a.grad[i, j] == ta.grad[i, j]
            assert b.grad[i, j] == tb.grad[i, j]