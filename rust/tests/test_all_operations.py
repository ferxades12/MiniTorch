import pytest
import numpy as np
import torch
import rustorch as rt

    
class TestAllOperations:
    """Test suite completo para todas las operaciones con casos edge y de error"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== POWER OPERATION TESTS ====================
    
    @pytest.mark.parametrize("base, exp", [
        # Casos normales
        ([[1, 2, 3]], [[0.5, 1.5, 2]]),
        ([[2, 3, 4]], [[1, 1, 1]]),
        ([[2, 3, 4]], 2),  # broadcasting
        
        # Casos edge
        ([[1]], [[0]]),  # 1^0 = 1
        ([[0]], [[1]]),  # 0^1 = 0
        ([[1e-7]], [[2]]),  # números muy pequeños
        ([[100]], [[0.1]]),  # números grandes con exp pequeño
        
        # Broadcasting cases
        ([[1, 2], [3, 4]], [[2]]),  # matriz ^ vector
        ([[2]], [[1, 2, 3]]),  # vector ^ matriz
    ])
    def test_pow(self, base, exp):
        """Test operación potencia (pow) - forward y backward"""
        A_rust = rt.Tensor(np.array(base, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(exp, dtype=np.float32), requires_grad=True) if not np.isscalar(exp) else exp
        
        A_torch = torch.tensor(base, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(exp, dtype=torch.float32, requires_grad=True) if not np.isscalar(exp) else exp
        
        C_rust = A_rust ** B_rust
        C_torch = A_torch ** B_torch
        
        # Forward pass
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-5)
        
        # Backward pass
        C_rust.sum().backward()
        C_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)
        if isinstance(B_rust, rt.Tensor):
            assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-5)

    # ==================== MULTIPLICATION TESTS ====================
    
    @pytest.mark.parametrize("a, b, case_type", [
        # Sin broadcasting
        ([[1, 2, 3]], [[4, 5, 6]], "same_shape"),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], "same_shape"),
        
        # Broadcasting válido
        ([[1, 2, 3], [4, 5, 6]], [10, 20, 30], "broadcast_row"),
        ([[1, 2, 3], [4, 5, 6]], [[10], [20]], "broadcast_col"),
        ([[1, 2, 3]], 5, "broadcast_scalar"),
        ([[[1, 2]]], [[1], [2]], "broadcast_3d"),
        
        # Casos edge
        ([[0, 1, -1]], [[2, -2, 3]], "with_zeros_negatives"),
        ([[1e-7, 1e7]], [[1e7, 1e-7]], "extreme_values"),
        ([[np.inf]], [[1]], "with_inf"),
        ([[1]], [[np.nan]], "with_nan"),
        
        # Casos de precisión
        ([[1.0000001]], [[1.0000002]], "precision"),
    ])
    def test_mul(self, a, b, case_type):
        """Test multiplicación elemento a elemento - forward y backward"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True) if not np.isscalar(b) else b
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C_rust = A_rust * B_rust
        C_torch = A_torch * B_torch
        
        # Skip backward si hay nan/inf
        if not (np.isnan(C_rust.numpy()).any() or np.isinf(C_rust.numpy()).any()):
            # Forward pass
            tolerance = 1e-4 if case_type in ["extreme_values", "precision"] else 1e-5
            assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=tolerance)
            
            # Backward pass
            C_rust.sum().backward()
            C_torch.sum().backward()
            
            assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)
            if isinstance(B_rust, rt.Tensor):
                assert B_rust.grad.shape == B_torch.grad.shape
                assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=tolerance)

    # ==================== TRANSPOSE TESTS ====================
    
    @pytest.mark.parametrize("arr, case_type", [
        # Casos normales
        ([[1, 2, 3], [4, 5, 6]], "normal_2x3"),
        ([[1, 2], [3, 4], [5, 6]], "normal_3x2"),
        
        # Casos edge
        ([[1]], "single_element"),
        ([[1, 2, 3, 4]], "single_row"),
        ([[1], [2], [3]], "single_col"),
        
        # Casos extremos
        (np.random.uniform(-1e6, 1e6, (100, 50)), "large_matrix"),
        ([[1e-10, 1e10]], "extreme_values"),
        ([[0, -0]], "zeros"),
        
        # Casos de precisión
        ([[1.0000001, 1.0000002]], "precision"),
    ])
    def test_transpose(self, arr, case_type):
        """Test transposición de matrices - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.t()
        B_torch = A_torch.t()
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["extreme_values", "precision", "large_matrix"] else 1e-5
        assert B_rust.numpy().shape == B_torch.detach().numpy().shape
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)

    # ==================== DOT PRODUCT / MATRIX MULTIPLICATION TESTS ====================
    
    @pytest.mark.parametrize("a, b", [
        # Casos válidos
        ([[1, 2, 3]], [4, 5, 6]),  # vector @ vector
        ([[1, 2]], [[3], [4]]),  # matriz @ vector
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),  # matriz @ matriz
        
        # Casos edge válidos
        ([[1]], [2]),
        ([1], [2]),
        
        # Casos extremos válidos
        (np.random.uniform(-5, 5, (2, 100)), np.random.uniform(-2, 2, (100, 3))),
        ([[1e-7, 1e7]], [[1e7], [1e-7]]),
    ])
    def test_dot(self, a, b):
        """Test producto punto / multiplicación matricial - forward y backward"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True)
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        C_rust = A_rust.dot(B_rust)
        C_torch = A_torch @ B_torch
        
        # Forward pass
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=1e-4)
        
        # Backward pass
        C_rust.sum().backward()
        C_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-4)
        assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=1e-4)

    # ==================== LOGARITHM TESTS ====================
    
    @pytest.mark.parametrize("arr", [
        # Casos válidos
        [1, 2, 3],
        [0.1, 1, 10],
        [1e-7, 1, 1e7],  # valores extremos pero > 0
        [1e-10],  # muy pequeño pero > 0
        [1e10],   # muy grande
    ])
    def test_log(self, arr):
        """Test logaritmo natural - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.log()
        B_torch = torch.log(A_torch)
        
        # Forward pass
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== ADDITION TESTS ====================
    
    @pytest.mark.parametrize("a, b, case_type", [
        # Broadcasting válido
        ([[1, 2, 3], [4, 5, 6]], [10, 20, 30], "broadcast_row"),
        ([[1, 2, 3], [4, 5, 6]], [[10], [20]], "broadcast_col"),
        ([[1, 2, 3]], 5, "broadcast_scalar"),
        
        # Sin broadcasting
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], "same_shape"),
        
        # Casos edge
        ([[0]], [[0]], "zeros"),
        ([[np.inf]], [[1]], "with_inf"),
        ([[1]], [[np.nan]], "with_nan"),
        ([[-1e10]], [[1e10]], "extreme_values"),
        
        # Casos de precisión
        ([[1.0000001]], [[1.0000002]], "precision"),
        
        # Broadcasting complejo
        ([[[1]]], [[1, 2]], "3d_broadcast"),
    ])
    def test_add(self, a, b, case_type):
        """Test suma - forward y backward"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True) if not np.isscalar(b) else b
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C_rust = A_rust + B_rust
        C_torch = A_torch + B_torch
        
        # Skip backward si hay nan
        if not np.isnan(C_rust.numpy()).any():
            # Forward pass
            tolerance = 1e-4 if case_type in ["extreme_values", "precision"] else 1e-5
            assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=tolerance)
            
            # Backward pass
            C_rust.sum().backward()
            C_torch.sum().backward()
            
            assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)
            if isinstance(B_rust, rt.Tensor):
                assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=tolerance)

    # ==================== SUBTRACTION TESTS ====================
    
    @pytest.mark.parametrize("a, b, case_type", [
        # Casos normales
        ([[5, 6, 7]], [[1, 2, 3]], "normal"),
        ([[10, 20], [30, 40]], [[5, 10], [15, 20]], "normal_matrix"),
        
        # Broadcasting
        ([[5, 6, 7]], [2], "broadcast_scalar"),
        ([[10, 20], [30, 40]], [[5], [10]], "broadcast_col"),
        
        # Casos edge
        ([[0]], [[0]], "zeros"),
        ([[1, 2]], [[-1, -2]], "with_negatives"),
        ([[-10, -20]], [[10, 20]], "negative_minus_positive"),
        
        # Casos extremos
        ([[1e10]], [[1e-10]], "extreme_subtraction"),
        ([[1.0000001]], [[1.0000000]], "precision"),
    ])
    def test_sub(self, a, b, case_type):
        """Test resta - forward y backward"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True) if not np.isscalar(b) else b
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C_rust = A_rust - B_rust
        C_torch = A_torch - B_torch
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["extreme_subtraction", "precision"] else 1e-5
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        C_rust.sum().backward()
        C_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)
        if isinstance(B_rust, rt.Tensor):
            assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=tolerance)

    # ==================== DIVISION TESTS ====================
    
    @pytest.mark.parametrize("a, b, case_type", [
        # Casos normales
        ([[4, 6, 8]], [[2, 2, 2]], "normal"),
        ([[1, 4, 9]], [[1, 2, 3]], "normal"),
        
        # Broadcasting
        ([[4, 6, 8]], [2], "broadcast"),
        ([[4, 6], [8, 10]], [[2], [5]], "broadcast_col"),
        
        # Casos con divisores negativos
        ([[1, 2]], [[-1, 2]], "negative_divisor"),
        
        # Casos extremos
        ([[1e10]], [[1e-10]], "extreme_division"),
        ([[1e-10]], [[1e10]], "tiny_dividend"),
    ])
    def test_div(self, a, b, case_type):
        """Test división - forward y backward"""
        A_rust = rt.Tensor(np.array(a, dtype=np.float32), requires_grad=True)
        B_rust = rt.Tensor(np.array(b, dtype=np.float32), requires_grad=True) if not np.isscalar(b) else b
        
        A_torch = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        B_torch = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C_rust = A_rust / B_rust
        C_torch = A_torch / B_torch
        
        # Forward pass
        tolerance = 1e-3 if case_type in ["extreme_division", "tiny_dividend"] else 1e-5
        assert np.allclose(C_rust.numpy(), C_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        C_rust.sum().backward()
        C_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)
        if isinstance(B_rust, rt.Tensor):
            assert np.allclose(B_rust.grad, B_torch.grad.numpy(), atol=tolerance)

    # ==================== SUM TESTS ====================
    
    @pytest.mark.parametrize("arr, axis, case_type", [
        # Casos normales
        ([[1, 2, 3], [4, 5, 6]], None, "sum_all"),
        ([[1, 2, 3], [4, 5, 6]], 0, "sum_axis_0"),
        ([[1, 2, 3], [4, 5, 6]], 1, "sum_axis_1"),
        
        # Casos edge
        ([[1]], None, "single_element"),
        ([1, 2, 3], None, "1d_array"),
        ([[[1, 2]], [[3, 4]]], None, "3d_array"),
        
        # Casos extremos
        ([[1e10, -1e10]], None, "large_numbers"),
        ([[0, 0, 0]], None, "all_zeros"),
        ([[np.inf, 1]], None, "with_inf"),
    ])
    def test_sum(self, arr, axis, case_type):
        """Test suma de elementos - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if axis is not None:
            B_rust = A_rust.sum(axis=axis)
            B_torch = A_torch.sum(dim=axis)
        else:
            B_rust = A_rust.sum()
            B_torch = A_torch.sum()
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["large_numbers"] else 1e-5
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        if B_rust.numdims() == 0:
            B_rust.backward()
            B_torch.backward()
        else:
            B_rust.sum().backward()
            B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)

    # ==================== ABSOLUTE VALUE TESTS ====================
    
    @pytest.mark.parametrize("arr", [
        [1, -2, 0, 3.5, -4.2],                # valores mixtos
        [[-1, 2], [-3, 0]],                   # matriz 2x2
        [[1e-7, -1e7]],                       # extremos
        [[0]],                                # solo cero
        np.random.uniform(-100, 100, (10,)),  # vector aleatorio
    ])
    def test_abs(self, arr):
        """Test valor absoluto - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B_rust = A_rust.abs()
        B_torch = A_torch.abs()

        # Forward pass
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-6)

        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()

        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-6)

    # ==================== EXPONENTIAL TESTS ====================
    
    @pytest.mark.parametrize("arr, case_type", [
        # Casos normales
        ([0, 1, 2], "normal"),
        ([[-1, 0, 1]], "with_negatives"),
        
        # Casos edge
        ([0], "zero"),
        ([-10, -5, 0, 5, 10], "range"),
        
        # Casos extremos
        ([1e-7], "very_small"),
        ([-20], "large_negative"),  # exp(-20) es muy pequeño
        ([10], "large_positive"),
        
        # Casos de precisión
        ([0.0000001], "precision"),
    ])
    def test_exp(self, arr, case_type):
        """Test función exponencial - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.exp()
        B_torch = torch.exp(A_torch)
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["large_positive", "precision"] else 1e-5
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)

    # ==================== SQUARE ROOT TESTS ====================
    
    @pytest.mark.parametrize("arr", [
        # Casos válidos
        [1, 4, 9],
        [0.25, 1, 2.25],
        [1e-7, 1, 1e7],
        [0],  # sqrt(0) = 0
        [1e-10],  # muy pequeño
    ])
    def test_sqrt(self, arr):
        """Test raíz cuadrada - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B_rust = A_rust.sqrt()
        B_torch = torch.sqrt(A_torch)
        
        # Forward pass
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
        
        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== MEAN TESTS ====================
    
    @pytest.mark.parametrize("arr, axis, case_type", [
        # Casos normales
        ([[1, 2, 3], [4, 5, 6]], None, "mean_all"),
        ([[1, 2, 3], [4, 5, 6]], 0, "mean_axis_0"),
        ([[1, 2, 3], [4, 5, 6]], 1, "mean_axis_1"),
        
        # Casos edge
        ([[1]], None, "single_element"),
        ([1, 2, 3, 4, 5], None, "1d_array"),
        
        # Casos extremos
        ([[1e10, -1e10]], None, "large_numbers"),
        ([[0, 0, 0]], None, "all_zeros"),
    ])
    def test_mean(self, arr, axis, case_type):
        """Test promedio - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if axis is not None:
            B_rust = A_rust.mean(axis=axis)
            B_torch = A_torch.mean(dim=axis)
        else:
            B_rust = A_rust.mean()
            B_torch = A_torch.mean()
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["large_numbers"] else 1e-5
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        if B_rust.numdims() == 0:
            B_rust.backward()
            B_torch.backward()
        else:
            B_rust.sum().backward()
            B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)

    # ==================== MAX TESTS ====================
    
    @pytest.mark.parametrize("arr, axis, case_type", [
        # Casos normales
        ([[1, 2, 3], [4, 5, 6]], None, "max_all"),
        ([[1, 2, 3], [4, 5, 6]], 0, "max_axis_0"),
        ([[1, 2, 3], [4, 5, 6]], 1, "max_axis_1"),
        
        # Casos edge
        ([[1]], None, "single_element"),
        ([1, 5, 3, 2, 4], None, "1d_array"),
        
        # Casos extremos
        ([[-1e10, 1e10]], None, "large_numbers"),
        ([[0, 0, 0]], None, "all_same"),
        ([[-5, -3, -10]], None, "all_negative"),
    ])
    def test_max(self, arr, axis, case_type):
        """Test máximo - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if axis is not None:
            B_rust = A_rust.max(axis=axis)
            B_torch = A_torch.max(dim=axis).values
        else:
            B_rust = A_rust.max()
            B_torch = A_torch.max()
        
        # Forward pass
        tolerance = 1e-4 if case_type in ["large_numbers"] else 1e-5
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=tolerance)
        
        # Backward pass
        if B_rust.numdims() == 0:
            B_rust.backward()
            B_torch.backward()
        else:
            B_rust.sum().backward()
            B_torch.sum().backward()
        
        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=tolerance)

    # ==================== RESHAPE TESTS ====================
    
    @pytest.mark.parametrize("arr, new_shape, should_fail", [
        # Casos válidos
        ([[1, 2, 3, 4]], (2, 2), False),
        ([[1, 2], [3, 4]], (4,), False),
        ([1, 2, 3, 4, 5, 6], (2, 3), False),
        ([1, 2, 3, 4, 5, 6], (3, 2), False),
        
        # Casos con -1 (inferir dimensión)
        ([1, 2, 3, 4, 5, 6], (-1, 2), False),
        ([1, 2, 3, 4, 5, 6], (2, -1), False),
        
        # Casos que fallan
        ([1, 2, 3], (2, 2), True),  # tamaño incompatible
        ([1, 2, 3, 4], (3, 2), True),  # tamaño incompatible
    ])
    def test_reshape(self, arr, new_shape, should_fail):
        """Test reshape - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if should_fail:
            with pytest.raises((ValueError, RuntimeError)):
                B_rust = A_rust.reshape(new_shape)
            with pytest.raises((ValueError, RuntimeError)):
                B_torch = A_torch.reshape(new_shape)
        else:
            B_rust = A_rust.reshape(new_shape)
            B_torch = A_torch.reshape(new_shape)
            
            # Forward pass
            assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-5)
            
            # Backward pass
            B_rust.sum().backward()
            B_torch.sum().backward()
            
            assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-5)

    # ==================== NEGATIVE TESTS ====================
    
    @pytest.mark.parametrize("arr", [
        [1, -2, 0, 3.5],
        [[-1, 2], [-3, 4]],
        [[1e-7, -1e7]],
        [[0]],
    ])
    def test_neg(self, arr):
        """Test negación - forward y backward"""
        A_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        A_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B_rust = -A_rust
        B_torch = -A_torch

        # Forward pass
        assert np.allclose(B_rust.numpy(), B_torch.detach().numpy(), atol=1e-6)

        # Backward pass
        B_rust.sum().backward()
        B_torch.sum().backward()

        assert np.allclose(A_rust.grad, A_torch.grad.numpy(), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
