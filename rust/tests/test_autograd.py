"""
Test suite para autograd del backend de Rust (rustorch)
Verifica el correcto funcionamiento del grafo computacional y backward pass
"""

import pytest
import numpy as np
import torch
import rustorch as rt


class TestAutograd:
    """Test suite para el sistema de autograd"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== BASIC AUTOGRAD ====================
    
    @pytest.mark.parametrize("arr,requires_grad", [
        ([[1, 2], [3, 4]], False),
        ([[1, 2], [3, 4]], True),
        ([[5, 10]], False),
        ([[5, 10]], True),
    ])
    def test_leaf_tensors(self, arr, requires_grad):
        """Test que los tensors iniciales son hojas"""
        A = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=requires_grad)
        
        assert A.is_leaf
        assert A.requires_grad == requires_grad

    def test_non_leaf_tensors(self):
        """Test que los resultados de operaciones no son hojas"""
        a = [[1, 2], [3, 4]]
        b = [[2, 3], [4, 5]]
        
        A = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        C = A + B
        D = A * B
        
        assert not C.is_leaf
        assert not D.is_leaf
        assert C.requires_grad
        assert D.requires_grad

    @pytest.mark.parametrize("a", [
        [[2, 3]],
        [[1, 2, 3]],
        [[1, 2], [3, 4]],
        [[5]],
    ])
    def test_gradient_propagation_simple(self, a):
        """Test propagación simple de gradientes"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        # Forward: y = sum(A)
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        # Backward
        B_rust.backward()
        B_torch.backward()
        
        # dy/dA = 1 para cada elemento
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a", [
        [[1, 2], [3, 4]],
        [[5, 10]],
        [[1, 2, 3], [4, 5, 6]],
    ])
    def test_gradient_propagation_chain(self, a):
        """Test propagación en cadena: y = sum(A + A)"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        # Forward
        B_rust = A_rust + A_rust
        B_torch = A_torch + A_torch
        
        C_rust = B_rust.sum()
        C_torch = B_torch.sum()
        
        # Backward
        C_rust.backward()
        C_torch.backward()
        
        # dy/dA = 2 (porque A aparece dos veces)
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== GRADIENT COMPUTATION ====================
    
    @pytest.mark.parametrize("a,b", [
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]]),
        ([[5, 10]], [[15, 20]]),
        ([[1, 2, 3]], [[4, 5, 6]]),
    ])
    def test_add_gradients(self, a, b):
        """Test gradientes de suma: d(A+B)/dA = 1, d(A+B)/dB = 1"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        C_rust = (A_rust + B_rust).sum()
        C_torch = (A_torch + B_torch).sum()
        
        C_rust.backward()
        C_torch.backward()
        
        # Los gradientes de suma son 1
        assert np.allclose(A_rust.grad, np.ones_like(a, dtype=np.float32), atol=1e-5)
        assert np.allclose(B_rust.grad, np.ones_like(b, dtype=np.float32), atol=1e-5)
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b", [
        ([[2, 3]], [[4, 5]]),
        ([[1, 2, 3]], [[2, 3, 4]]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ])
    def test_mul_gradients(self, a, b):
        """Test gradientes de multiplicación: d(A*B)/dA = B, d(A*B)/dB = A"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        C_rust = (A_rust * B_rust).sum()
        C_torch = (A_torch * B_torch).sum()
        
        C_rust.backward()
        C_torch.backward()
        
        # d/dA = B, d/dB = A
        assert np.allclose(A_rust.grad, b, atol=1e-5)
        assert np.allclose(B_rust.grad, a, atol=1e-5)
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a", [
        [[1, 2, 3], [4, 5, 6]],
        [[5, 10]],
        [[1, 2, 3]],
    ])
    def test_sum_gradients(self, a):
        """Test gradientes de sum: d(sum(A))/dA = 1s"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        B_rust.backward()
        B_torch.backward()
        
        # Todos los gradientes son 1
        assert np.allclose(A_rust.grad, np.ones_like(a, dtype=np.float32), atol=1e-5)
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== COMPLEX COMPUTATIONAL GRAPHS ====================
    
    @pytest.mark.parametrize("a,b,c", [
        ([[1, 2]], [[3, 4]], [[5, 6]]),
        ([[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]),
        ([[2]], [[4]], [[6]]),
    ])
    def test_complex_graph_1(self, a, b, c):
        """Test grafo complejo: y = sum((A + B) * (A + C))"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        C_rust = rt.Tensor(np.array(c, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        C_torch = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        
        # Forward
        Y_rust = ((A_rust + B_rust) * (A_rust + C_rust)).sum()
        Y_torch = ((A_torch + B_torch) * (A_torch + C_torch)).sum()
        
        assert np.allclose(Y_rust.numpy(), Y_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        Y_rust.backward()
        Y_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-4)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(C_rust.grad, C_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b,c", [
        ([[2, 3], [4, 5]], [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 2]], [[3, 4]], [[5, 6]]),
        ([[5]], [[10]], [[15]]),
    ])
    def test_complex_graph_2(self, a, b, c):
        """Test grafo complejo: y = sum(A * B + A * C)"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        C_rust = rt.Tensor(np.array(c, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        C_torch = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        
        # Forward
        Y_rust = (A_rust * B_rust + A_rust * C_rust).sum()
        Y_torch = (A_torch * B_torch + A_torch * C_torch).sum()
        
        assert np.allclose(Y_rust.numpy(), Y_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        Y_rust.backward()
        Y_torch.backward()
        
        # d/dA = B + C (por la regla de la suma)
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(C_rust.grad, C_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b", [
        ([[1, 2]], [[3, 4]]),
        ([[2, 3, 4]], [[5, 6, 7]]),
    ])
    def test_deep_chain(self, a, b):
        """Test cadena profunda de operaciones"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        # Forward: ((A + B) * A + B) * A
        C1_rust = A_rust + B_rust
        C1_torch = A_torch + B_torch
        
        C2_rust = C1_rust * A_rust
        C2_torch = C1_torch * A_torch
        
        C3_rust = C2_rust + B_rust
        C3_torch = C2_torch + B_torch
        
        C4_rust = C3_rust * A_rust
        C4_torch = C3_torch * A_torch
        
        Y_rust = C4_rust.sum()
        Y_torch = C4_torch.sum()
        
        assert np.allclose(Y_rust.numpy(), Y_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        Y_rust.backward()
        Y_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-4)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-4)

    # ==================== EDGE CASES ====================
    
    def test_no_grad_tensor_in_computation(self):
        """Test tensor sin gradiente en el grafo"""
        a = [[1, 2]]
        b = [[3, 4]]
        
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=False)
        
        C_rust = (A_rust + B_rust).sum()
        
        C_rust.backward()
        
        # Solo A debería tener gradiente
        assert A_rust.grad is not None
        assert B_rust.grad is None

    def test_scalar_backward(self):
        """Test backward desde un escalar"""
        a = [[5]]
        
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()  # Resultado escalar
        B_torch = A_torch.sum()
        
        # Backward sin pasar gradiente inicial
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    def test_multiple_uses_same_tensor(self):
        """Test usar el mismo tensor múltiples veces"""
        a = [[2, 3]]
        
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        # A se usa 3 veces
        B_rust = (A_rust * A_rust + A_rust).sum()
        B_torch = (A_torch * A_torch + A_torch).sum()
        
        B_rust.backward()
        B_torch.backward()
        
        # d/dA = 2*A + 1
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

