"""
Test suite para operaciones básicas del backend de Rust (rustorch)
Compara los resultados con PyTorch para verificar correctitud
"""

import pytest
import numpy as np
import torch
import rustorch as rt


class TestBasicOperations:
    """Test suite para operaciones básicas con el backend de Rust"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== ADDITION TESTS ====================
    
    @pytest.mark.parametrize("a, b", [
        # Casos normales
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        
        # Casos con ceros y negativos
        ([[0, 1, -1]], [[2, -2, 3]]),
        ([[-1, -2], [-3, -4]], [[1, 2], [3, 4]]),
        
        # Valores extremos
        ([[1e-7, 1e7]], [[1e7, 1e-7]]),
        ([[100, 200]], [[300, 400]]),
    ])
    def test_add_forward(self, a, b):
        """Test suma - forward pass"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=False)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=False)
        
        A_torch = torch.tensor(a, dtype=torch.float32)
        B_torch = torch.tensor(b, dtype=torch.float32)
        
        C_rust = A_rust + B_rust
        C_torch = A_torch + B_torch
        
        assert np.allclose(C_rust.numpy(), C_torch.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a, b", [
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[0.5, 1.5]], [[2.5, 3.5]]),
    ])
    def test_add_backward(self, a, b):
        """Test suma - backward pass"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        C_rust = A_rust + B_rust
        C_torch = A_torch + B_torch
        
        # Forward pass
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-5)
        
        # Backward pass
        D_rust = C_rust.sum()
        D_torch = C_torch.sum()
        
        D_rust.backward()
        D_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    # ==================== MULTIPLICATION TESTS ====================
    
    @pytest.mark.parametrize("a, b", [
        # Casos normales
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[2, 3], [4, 5]], [[1, 2], [3, 4]]),
        
        # Casos con ceros
        ([[0, 1, 2]], [[3, 4, 5]]),
        ([[1, 2, 3]], [[0, 0, 0]]),
        
        # Casos con negativos
        ([[-1, 2, -3]], [[4, -5, 6]]),
        
        # Valores extremos
        ([[1e-3, 1e3]], [[1e3, 1e-3]]),
    ])
    def test_mul_forward(self, a, b):
        """Test multiplicación - forward pass"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=False)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=False)
        
        A_torch = torch.tensor(a, dtype=torch.float32)
        B_torch = torch.tensor(b, dtype=torch.float32)
        
        C_rust = A_rust * B_rust
        C_torch = A_torch * B_torch
        
        assert np.allclose(C_rust.numpy(), C_torch.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a, b", [
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[2, 3], [4, 5]], [[1, 2], [3, 4]]),
        ([[0.5, 1.5, 2.5]], [[2, 3, 4]]),
    ])
    def test_mul_backward(self, a, b):
        """Test multiplicación - backward pass"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        C_rust = A_rust * B_rust
        C_torch = A_torch * B_torch
        
        # Forward pass
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-5)
        
        # Backward pass
        D_rust = C_rust.sum()
        D_torch = C_torch.sum()
        
        D_rust.backward()
        D_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    # ==================== COMBINED OPERATIONS TESTS ====================
    
    @pytest.mark.parametrize("a,b,c", [
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], [[0.5, 0.5], [0.5, 0.5]]),
        ([[1, 2, 3]], [[4, 5, 6]], [[0.1, 0.2, 0.3]]),
        ([[5]], [[10]], [[2]]),
    ])
    def test_combined_ops(self, a, b, c):
        """Test operaciones combinadas: (A + B) * C"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        C_rust = rt.Tensor(np.array(c, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        C_torch = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        
        # Forward: (A + B) * C
        D_rust = (A_rust + B_rust) * C_rust
        D_torch = (A_torch + B_torch) * C_torch
        
        assert np.allclose(D_rust.numpy(), D_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        E_rust = D_rust.sum()
        E_torch = D_torch.sum()
        
        E_rust.backward()
        E_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(C_rust.grad, C_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b,c", [
        ([[2, 3]], [[4, 5]], [[0.5, 0.5]]),
        ([[1, 2, 3]], [[2, 3, 4]], [[1, 1, 1]]),
        ([[5, 10]], [[2, 3]], [[0.1, 0.2]]),
    ])
    def test_chained_multiplications(self, a, b, c):
        """Test multiplicaciones encadenadas: A * B * C"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        C_rust = rt.Tensor(np.array(c, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        C_torch = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        
        # Forward: A * B * C
        D_rust = A_rust * B_rust * C_rust
        D_torch = A_torch * B_torch * C_torch
        
        assert np.allclose(D_rust.numpy(), D_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        E_rust = D_rust.sum()
        E_torch = D_torch.sum()
        
        E_rust.backward()
        E_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(C_rust.grad, C_torch.grad.numpy(), atol=1e-5)

    # ==================== EDGE CASES ====================
    
    @pytest.mark.parametrize("a,b,requires_grad_a,requires_grad_b", [
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], True, True),
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], False, False),
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], True, False),
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], False, True),
    ])
    def test_requires_grad_flag(self, a, b, requires_grad_a, requires_grad_b):
        """Test comportamiento del flag requires_grad"""
        A = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=requires_grad_a)
        B = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=requires_grad_b)
        C = A + B
        
        # El resultado requiere grad si alguno de los operandos lo requiere
        expected_requires_grad = requires_grad_a or requires_grad_b
        assert C.requires_grad == expected_requires_grad
        
        # Si no requiere grad, debería ser hoja
        if not expected_requires_grad:
            assert C.is_leaf
        else:
            assert not C.is_leaf
        
        assert A.is_leaf
        assert B.is_leaf

    def test_scalar_output(self):
        """Test operaciones que resultan en escalares"""
        a = [[1, 2], [3, 4]]
        
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        # Sum sin axis (total)
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward desde escalar (no necesita grad inicial)
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    def test_zero_gradients(self):
        """Test que los gradientes se acumulan correctamente desde cero"""
        a = [[1, 2], [3, 4]]
        b = [[2, 3], [4, 5]]
        
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        # Verificar que los gradientes iniciales son cero
        C_rust = A_rust + B_rust
        D_rust = C_rust.sum()
        D_rust.backward()
        
        # Los gradientes de suma deberían ser todos 1s
        expected_grad = np.ones_like(a, dtype=np.float32)
        assert np.allclose(A_rust.grad, expected_grad, atol=1e-5)
        assert np.allclose(B_rust.grad, expected_grad, atol=1e-5)

