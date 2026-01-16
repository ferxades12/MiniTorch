"""
Test suite para la función sum() del backend de Rust (rustorch)
Compara los resultados con PyTorch para verificar correctitud
"""

import pytest
import numpy as np
import torch
import rustorch as rt


class TestSumOperation:
    """Test suite para la operación sum con diferentes ejes"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== SUM TOTAL (sin axis) ====================
    
    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[0, 0, 0]],
        [[-1, 2, -3]],
        [[1.5, 2.5, 3.5]],
    ])
    def test_sum_total_forward(self, arr):
        """Test sum sin axis - forward pass"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=False)
        A_torch = torch.tensor(arr, dtype=torch.float32)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.numpy(), atol=1e-5)

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[5, 10, 15]],
    ])
    def test_sum_total_backward(self, arr):
        """Test sum sin axis - backward pass"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        # Forward
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        B_rust.backward()
        B_torch.backward()
        
        # El gradiente de sum total es un tensor de 1s del mismo shape
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(A_rust.grad, np.ones_like(arr, dtype=np.float32), atol=1e-5)

    # ==================== SUM CON AXIS ====================
    
    @pytest.mark.parametrize("arr, axis", [
        ([[1, 2, 3], [4, 5, 6]], 0),  # Suma por columnas
        ([[1, 2, 3], [4, 5, 6]], 1),  # Suma por filas
        ([[1, 2], [3, 4], [5, 6]], 0),  # 3x2 -> suma columnas
        ([[1, 2], [3, 4], [5, 6]], 1),  # 3x2 -> suma filas
    ])
    def test_sum_axis_forward(self, arr, axis):
        """Test sum con axis - forward pass"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=False)
        A_torch = torch.tensor(arr, dtype=torch.float32)
        
        B_rust = A_rust.sum(axis=axis)
        B_torch = A_torch.sum(dim=axis)
        
        assert np.allclose(B_rust.numpy(), B_torch.numpy(), atol=1e-5)

    @pytest.mark.parametrize("arr, axis", [
        ([[1, 2, 3], [4, 5, 6]], 0),
        ([[1, 2, 3], [4, 5, 6]], 1),
        ([[5, 10], [15, 20], [25, 30]], 0),
        ([[5, 10], [15, 20], [25, 30]], 1),
    ])
    def test_sum_axis_backward(self, arr, axis):
        """Test sum con axis - backward pass"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum(axis=axis)
        B_torch = A_torch.sum(dim=axis)
        
        # Forward
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward - necesitamos sum total para obtener escalar
        C_rust = B_rust.sum()
        C_torch = B_torch.sum()
        
        C_rust.backward()
        C_torch.backward()
        
        # Los gradientes deben propagarse correctamente
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== SUM EN OPERACIONES COMPUESTAS ====================
    
    @pytest.mark.parametrize("a,b", [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[1, 2]], [[3, 4]]),
        ([[5]], [[10]]),
    ])
    def test_sum_after_add(self, a, b):
        """Test sum después de suma: (A + B).sum()"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        # Forward
        C_rust = (A_rust + B_rust).sum()
        C_torch = (A_torch + B_torch).sum()
        
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        C_rust.backward()
        C_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b", [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[1, 2]], [[3, 4]]),
        ([[2, 3, 4]], [[5, 6, 7]]),
    ])
    def test_sum_after_mul(self, a, b):
        """Test sum después de multiplicación: (A * B).sum()"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        # Forward
        C_rust = (A_rust * B_rust).sum()
        C_torch = (A_torch * B_torch).sum()
        
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        C_rust.backward()
        C_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a", [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[5, 10]],
    ])
    def test_double_sum(self, a):
        """Test doble sum: A.sum(axis=1).sum()"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        # Forward
        B_rust = A_rust.sum(axis=1).sum()
        B_torch = A_torch.sum(dim=1).sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== EDGE CASES ====================
    
    @pytest.mark.parametrize("a", [
        [[5]],
        [[10]],
        [[3.14]],
    ])
    def test_sum_single_element(self, a):
        """Test sum de un solo elemento"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a", [
        [[0, 0, 0], [0, 0, 0]],
        [[0]],
        [[0, 0]],
    ])
    def test_sum_zeros(self, a):
        """Test sum de tensor de ceros"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a", [
        [[-1, -2, -3], [-4, -5, -6]],
        [[-10]],
        [[-1, -2, -3]],
    ])
    def test_sum_negative_values(self, a):
        """Test sum con valores negativos"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sum()
        B_torch = A_torch.sum()
        
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        B_rust.backward()
        B_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("a,b,c", [
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]], [[0.5, 0.5], [0.5, 0.5]]),
        ([[1, 2, 3]], [[2, 3, 4]], [[1, 1, 1]]),
        ([[5]], [[10]], [[2]]),
    ])
    def test_sum_mixed_operations(self, a, b, c):
        """Test sum en operaciones mixtas complejas"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        C_rust = rt.Tensor(np.array(c, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        C_torch = torch.tensor(c, dtype=torch.float32, requires_grad=True)
        
        # Forward: ((A + B) * C).sum()
        D_rust = ((A_rust + B_rust) * C_rust).sum()
        D_torch = ((A_torch + B_torch) * C_torch).sum()
        
        assert np.allclose(D_rust.numpy(), D_torch.detach().numpy(), atol=1e-5)
        
        # Backward
        D_rust.backward()
        D_torch.backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)
        assert np.allclose(C_rust.grad, C_torch.grad.numpy(), atol=1e-5)

