import src as M
import torch
import pytest
from src.activations import *
from src.losses import *
import numpy as np

@pytest.mark.parametrize("arr", [
    [0, 1, 2, 3],                # valores crecientes
    [-1, 0, 1],                  # negativos y positivos
    [1000, 1001, 1002],          # valores grandes (test de estabilidad numérica)
    [-1000, 0, 1000],            # extremos
    [0, 0, 0],                   # todos iguales
    np.random.uniform(-5, 5, 5), # aleatorio
])
def test_softmax_1d(arr):
    soft = Softmax()

    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = soft(A)
    tb = torch.nn.functional.softmax(ta, dim=0)

    B.sum().backward()
    tb.sum().backward()

    assert ta.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[1, 2, 3], [4, 5, 6]],  # Matriz 2x3
    [[1, 2], [3, 4], [5, 6]],  # Matriz 3x2
    [[1, 2, 3, 4]],  # Matriz 1x4
    np.random.uniform(-5, 5, (4, 2)),  # Matriz aleatoria 4x2
    [[7]],  # Matriz 1x1
])
def test_softmax(arr):
    soft = Softmax()

    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    # Aplana el tensor para aplicar softmax a todos los elementos
    A_flat = A.reshape((-1,))
    ta_flat = ta.view(-1)

    # Aplica softmax en MiniTorch y PyTorch
    B = soft(A_flat)
    tb = torch.nn.functional.softmax(ta_flat, dim=0)

    # Calcula el backward
    B.sum().backward()
    tb.sum().backward()

    # Verifica que los resultados sean iguales
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
    assert ta.grad is not None and np.allclose(A.grad, ta.grad, atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],           # negativos, ceros y positivos
    [[0, 0, 0]],                  # solo ceros
    [[-1000, 0, 1000]],           # extremos
    [[-1e-7, 0, 1e-7]],           # valores muy pequeños
    np.random.uniform(-5, 5, (3, 3)), # aleatorio
])
def test_relu(arr):
    relu = ReLU()

    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = relu(A)
    tb = torch.nn.functional.relu(ta)

    B.sum().backward()
    tb.sum().backward()

    assert ta.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],
    [[-10, -1, 0, 1, 10]],
    [[-100, -0.5, 0, 0.5, 100]],
    [[-2, -0.1, 0], [0.1, 2, 5]],
    np.random.uniform(-5, 5, (3, 3)),
])
def test_sigmoid(arr):
    sig = Sigmoid()

    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = sig(A)
    tb = torch.sigmoid(ta)

    B.sum().backward()
    tb.sum().backward()

    assert ta.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    [[-1, 0], [1, -1]],
    [[-10, -1, 0, 1, 10]],
    [[-100, -0.5, 0, 0.5, 100]],
    [[-2, -0.1, 0], [0.1, 2, 5]],
    np.random.uniform(-5, 5, (3, 3)),
])
def test_tanh(arr):
    tanh = Tanh()

    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    B = tanh(A)
    tb = torch.tanh(ta)

    B.sum().backward()
    tb.sum().backward()

    assert ta.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
    assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize("arr", [
    np.random.uniform(-2, 2, (1, 4)),           # batch size 1
    np.random.uniform(-2, 2, (8, 4)),           # batch grande
    np.array([[1e10, -1e10, 1e-10, -1e-10]]),   # valores extremos grandes y pequeños
    np.array([[1e-7, -1e-7, 1e-8, -1e-8], [1e7, -1e7, 1e8, -1e8]]), # mezcla extremos
    np.random.uniform(-1000, 1000, (3, 3)),     # rango muy amplio
])
def test_chain_activations(arr):
    tanh = Tanh()
    sigmoid = Sigmoid()
    relu = ReLU()
    softmax = Softmax()

    # Variables de entrada
    A = M.Tensor(arr, requires_grad=True)
    ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    # Pesos
    W1 = M.Tensor(np.random.randn(arr.shape[1], arr.shape[1]), requires_grad=True)
    w1 = torch.tensor(W1.data, dtype=torch.float32, requires_grad=True)

    W2 = M.Tensor(np.random.randn(arr.shape[1], arr.shape[1]), requires_grad=True)
    w2 = torch.tensor(W2.data, dtype=torch.float32, requires_grad=True)

    # Forward MiniTorch
    Z1 = A.dot(W1)
    H1 = tanh(Z1)
    Z2 = H1.dot(W2)
    H2 = sigmoid(Z2)
    H3 = relu(H2)
    Out = softmax(H3)
    loss = Out.sum()

    # Forward PyTorch
    z1_t = ta @ w1
    h1_t = torch.tanh(z1_t)
    z2_t = h1_t @ w2
    h2_t = torch.sigmoid(z2_t)
    h3_t = torch.relu(h2_t)
    out_t = torch.softmax(h3_t, dim=1)
    loss_t = out_t.sum()

    # Backward
    loss.backward()
    loss_t.backward()

    # Output final
    assert np.allclose(Out.data, out_t.detach().numpy(), atol=1e-5)

    # Gradientes del input
    assert ta.grad is not None and np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

    # Gradientes de pesos W1 y W2
    assert w1.grad is not None and np.allclose(W1.grad, w1.grad.numpy(), atol=1e-5)
    assert w2.grad is not None and np.allclose(W2.grad, w2.grad.numpy(), atol=1e-5)

    # Gradientes de nodos intermedios
    # Ejemplo: gradiente de Z1 y H1 respecto a la suma final
    if hasattr(Z1, "grad") and Z1.grad is not None:
        assert Z1.grad.shape == Z1.data.shape
    if hasattr(H1, "grad") and H1.grad is not None:
        assert H1.grad.shape == H1.data.shape

if __name__ == "__main__":
    pytest.main([__file__])