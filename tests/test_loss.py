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
    assert (ta.grad is not None) and np.allclose(a.grad, ta.grad.numpy(), atol=1e-5)

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
    assert (ta.grad is not None) and np.allclose(a.grad, ta.grad, atol=1e-5)

@pytest.mark.parametrize("arr", [
    np.random.uniform(-2, 2, (1, 4)),           # batch size 1
    np.random.uniform(-2, 2, (8, 4)),           # batch grande
    np.array([[1e-3, -1e-3, 1e-2, -1e-2]]),     # valores extremos pequeños
    np.random.uniform(-10, 10, (3, 3)),         # rango amplio
])
def test_chain(arr):
    # ========= Definición de pérdidas =========
    mse = MSE()
    ce = CrossEntropy()

    mse_torch = torch.nn.MSELoss()
    ce_torch = torch.nn.CrossEntropyLoss()

    # ========= Tensores de entrada =========
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    # Target para MSE (mismo shape que input)
    target_arr = np.random.uniform(-1, 1, arr.shape)
    T = M.Tensor(target_arr, requires_grad=False)
    tt = torch.tensor(target_arr, dtype=torch.float32, requires_grad=False)

    # Target para CrossEntropy (clases enteras por fila)
    # -> tomamos argmax de target_arr para simular etiquetas
    target_classes = np.random.randint(0, arr.shape[1], size=(arr.shape[0],))
    Tc = M.Tensor(target_classes, requires_grad=False)
    ttc = torch.tensor(target_classes, dtype=torch.long)

    # ========= Forward MSE =========
    loss_mse = mse(A, T)
    loss_mse_torch = mse_torch(ta, tt)

    # ========= Forward CrossEntropy =========
    # Para CE, interpretamos A como logits (no softmax)
    loss_ce = ce(A, Tc)
    loss_ce_torch = ce_torch(ta, ttc)

    # ========= Backward =========
    loss_mse.backward()
    loss_mse_torch.backward(retain_graph=True)  # para no perder gradientes de ta
    loss_ce.backward()
    loss_ce_torch.backward()

    # ========= Comprobaciones =========
    # MSE outputs
    assert np.allclose(loss_mse.data, loss_mse_torch.detach().numpy(), atol=1e-5)

    # CE outputs
    assert np.allclose(loss_ce.data, loss_ce_torch.detach().numpy(), atol=1e-5)

    # Gradientes input para MSE (la suma de backward se acumula)
    assert ta.grad is not None
    assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

    # ========= Comprobación de gradientes de nodos intermedios =========
    # (Si tu framework expone gradientes intermedios)
    if hasattr(A, "grad") and A.grad is not None:
        assert A.grad.shape == A.data.shape


if __name__ == "__main__":
    pytest.main([__file__])