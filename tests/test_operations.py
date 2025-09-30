import src as M
import torch
import pytest
import numpy as np
from src.nn.activations import *
from src.nn.losses import *

class TestOperations:
    """Test suite para todas las operaciones con casos edge y de error"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    @pytest.mark.parametrize("base, exp, should_fail", [
        # Casos normales
        ([[1, 2, 3]], [[0.5, 1.5, 2]], False),
        ([[2, 3, 4]], [[1, 1, 1]], False),
        ([[2, 3, 4]], 2, False),  # broadcasting
        
        # Casos edge
        ([[1]], [[0]], False),  # 1^0 = 1
        ([[0]], [[1]], False),  # 0^1 = 0
        ([[1e-7]], [[2]], False),  # números muy pequeños
        ([[100]], [[0.1]], False),  # números grandes con exp pequeño
        
        # Casos problemáticos (pueden dar nan/inf)
        ([[0]], [[0]], True),   # 0^0 = indeterminado
        ([[-1]], [[0.5]], True),  # (-1)^0.5 = nan
        ([[0]], [[-1]], True),   # 0^(-1) = inf
        
        # Broadcasting cases
        ([[1, 2], [3, 4]], [[2]], False),  # matriz ^ vector
        ([[2]], [[1, 2, 3]], False),  # vector ^ matriz
    ])
    def test_pow(self, base, exp, should_fail):
        A = M.Tensor(base, requires_grad=True)
        B = M.Tensor(exp, requires_grad=True) if not np.isscalar(exp) else exp
        
        ta = torch.tensor(base, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(exp, dtype=torch.float32, requires_grad=True) if not np.isscalar(exp) else exp
        
        if should_fail:
            with pytest.warns(RuntimeWarning):  # NumPy warnings para nan/inf
                C = A ** B
                tc = ta ** tb
                # No hacer backward si hay nan/inf
                if not (np.isnan(C.data).any() or np.isinf(C.data).any()):
                    C.sum().backward()
                    tc.sum().backward()
        else:
            C = A ** B
            tc = ta ** tb
            
            C.sum().backward()
            tc.sum().backward()
            
            assert np.allclose(C.data, tc.detach().numpy(), atol=1e-5)
            assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
            if isinstance(B, M.Tensor):
                assert np.allclose(B.grad, tb.grad.numpy(), atol=1e-5)

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
        A = M.Tensor(a, requires_grad=True)
        B = M.Tensor(b, requires_grad=True) if not np.isscalar(b) else b
        
        ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C = A * B
        tc = ta * tb
        
        # Skip backward si hay nan/inf
        if not (np.isnan(C.data).any() or np.isinf(C.data).any()):
            C.sum().backward()
            tc.sum().backward()
            
            tolerance = 1e-4 if case_type in ["extreme_values", "precision"] else 1e-5
            
            assert np.allclose(C.data, tc.detach().numpy(), atol=tolerance)
            assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)
            if isinstance(B, M.Tensor):
                assert B.grad.shape == tb.grad.shape
                assert np.allclose(B.grad, tb.grad.numpy(), atol=tolerance)

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
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B = A.T()
        tb = ta.t()
        
        B.sum().backward()
        tb.sum().backward()
        
        tolerance = 1e-4 if case_type in ["extreme_values", "precision", "large_matrix"] else 1e-5
        
        assert B.data.shape == tb.detach().numpy().shape
        assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)

    @pytest.mark.parametrize("a, b, should_fail", [
        # Casos válidos
        ([1, 2, 3], [4, 5, 6], False),  # vector @ vector
        ([[1, 2]], [[3], [4]], False),  # matriz @ vector
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], False),  # matriz @ matriz
        
        # Casos edge válidos
        ([[1]], [[2]], False),  # 1x1 @ 1x1
        ([1], [2], False),  # escalar @ escalar (como vectores)
        
        # Casos que fallan por dimensiones incompatibles
        ([[1, 2]], [[3, 4]], True),  # 1x2 @ 1x2 (incompatible)
        ([[1, 2], [3, 4]], [5, 6, 7], True),  # 2x2 @ 3 (incompatible)
        
        # Casos extremos válidos
        (np.random.uniform(-5, 5, (2, 100)), np.random.uniform(-2, 2, (100, 3)), False),
        ([[1e-7, 1e7]], [[1e7], [1e-7]], False),
    ])
    def test_dot(self, a, b, should_fail):
        A = M.Tensor(a, requires_grad=True)
        B = M.Tensor(b, requires_grad=True)
        
        ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        
        if should_fail:
            with pytest.raises((ValueError, RuntimeError)):
                C = A.dot(B)
            with pytest.raises((ValueError, RuntimeError)):
                tc = ta @ tb
        else:
            C = A.dot(B)
            tc = ta @ tb
            
            C.sum().backward()
            tc.sum().backward()
            
            assert np.allclose(C.data, tc.detach().numpy(), atol=1e-4)
            assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-4)
            assert np.allclose(B.grad, tb.grad.numpy(), atol=1e-4)

    @pytest.mark.parametrize("arr, should_fail", [
        # Casos válidos
        ([1, 2, 3], False),
        ([0.1, 1, 10], False),
        ([1e-7, 1, 1e7], False),  # valores extremos pero > 0
        
        # Casos que causan problemas
        ([0], True),  # log(0) = -inf
        ([-1], True),  # log(-1) = nan
        ([0, 1, 2], True),  # mezcla con 0
        ([-1, 0, 1], True),  # mezcla con negativos y 0
        
        # Casos edge válidos
        ([1e-10], False),  # muy pequeño pero > 0
        ([1e10], False),   # muy grande
    ])
    def test_log(self, arr, should_fail):
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if should_fail:
            with pytest.warns(RuntimeWarning):  # NumPy warning para log de negativos/cero
                B = A.log()
                tb = torch.log(ta)
                # No hacer backward si hay nan/inf
                if not (np.isnan(B.data).any() or np.isinf(B.data).any()):
                    B.sum().backward()
                    tb.sum().backward()
        else:
            B = A.log()
            tb = torch.log(ta)
            
            B.sum().backward()
            tb.sum().backward()
            
            assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
            assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

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
        A = M.Tensor(a, requires_grad=True)
        B = M.Tensor(b, requires_grad=True) if not np.isscalar(b) else b
        
        ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        C = A + B
        tc = ta + tb
        
        # Skip backward si hay nan
        if not np.isnan(C.data).any():
            C.sum().backward()
            tc.sum().backward()
            
            tolerance = 1e-4 if case_type in ["extreme_values", "precision"] else 1e-5
            
            assert np.allclose(C.data, tc.detach().numpy(), atol=tolerance)
            assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)
            if isinstance(B, M.Tensor):
                assert np.allclose(B.grad, tb.grad.numpy(), atol=tolerance)

    @pytest.mark.parametrize("a, b, case_type", [
        # Casos normales
        ([[4, 6, 8]], [[2, 2, 2]], "normal"),
        ([[1, 4, 9]], [[1, 2, 3]], "normal"),
        
        # Broadcasting
        ([[4, 6, 8]], [2], "broadcast"),
        ([[4, 6], [8, 10]], [[2], [5]], "broadcast_col"),
        
        # Casos problemáticos
        ([[1, 2]], [[0, 1]], "division_by_zero"),
        ([[1, 2]], [[-1, 2]], "negative_divisor"),
        ([[0]], [[0]], "zero_div_zero"),
        
        # Casos extremos
        ([[1e10]], [[1e-10]], "extreme_division"),
        ([[1e-10]], [[1e10]], "tiny_dividend"),
    ])
    def test_div(self, a, b, case_type):
        A = M.Tensor(a, requires_grad=True)
        B = M.Tensor(b, requires_grad=True) if not np.isscalar(b) else b
        
        ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True) if not np.isscalar(b) else b
        
        if case_type in ["division_by_zero", "zero_div_zero"]:
            with pytest.warns(RuntimeWarning):  # División por cero
                C = A / B
                tc = ta / tb
        else:
            C = A / B
            tc = ta / tb
            
            # Skip backward si hay inf/nan
            if not (np.isnan(C.data).any() or np.isinf(C.data).any()):
                C.sum().backward()
                tc.sum().backward()
                
                tolerance = 1e-3 if case_type in ["extreme_division", "tiny_dividend"] else 1e-5
                
                assert np.allclose(C.data, tc.detach().numpy(), atol=tolerance)
                assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)
                if isinstance(B, M.Tensor):
                    assert np.allclose(B.grad, tb.grad.numpy(), atol=tolerance)

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
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        if axis is not None:
            B = A.sum(axis=axis)
            tb = ta.sum(dim=axis)
        else:
            B = A.sum()
            tb = ta.sum()
        
        B.backward() if B.numdims() == 0 else B.sum().backward()
        tb.backward() if tb.numel() == 1 else tb.sum().backward()
        
        tolerance = 1e-4 if case_type in ["large_numbers"] else 1e-5
        
        assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

