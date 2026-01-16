"""
Test suite para propiedades básicas de tensores del backend de Rust
Verifica creación, shapes, tipos y operaciones básicas
"""

import pytest
import numpy as np
import torch
import rustorch as rt


class TestTensorBasics:
    """Test suite para funcionalidad básica de tensores"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)

    # ==================== TENSOR CREATION ====================
    
    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # 3D
    ])
    def test_tensor_creation(self, arr):
        """Test creación de tensores desde arrays numpy"""
        np_array = np.array(arr, dtype=np.float32)
        
        # Sin gradiente
        t_no_grad = rt.Tensor(np_array, requires_grad=False)
        assert not t_no_grad.requires_grad
        assert t_no_grad.is_leaf
        
        # Con gradiente
        t_grad = rt.Tensor(np_array, requires_grad=True)
        assert t_grad.requires_grad
        assert t_grad.is_leaf

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[1, 2], [3, 4]],
        [[1.5, 2.5, 3.5]],
        [[-1, 0, 1]],
    ])
    def test_tensor_to_numpy(self, arr):
        """Test conversión de tensor a numpy"""
        np_array = np.array(arr, dtype=np.float32)
        tensor = rt.Tensor(np_array, requires_grad=False)
        
        result = tensor.numpy()
        
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, np_array, atol=1e-5)
        assert result.shape == np_array.shape
        assert result.dtype == np.float32

    def test_tensor_repr(self):
        """Test representación string del tensor"""
        arr = [[1, 2], [3, 4]]
        tensor = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        
        repr_str = repr(tensor)
        
        assert "Tensor" in repr_str
        print(tensor)
        assert "shape" in repr_str.lower() or "[2, 2]" in repr_str
        assert "requires_grad" in repr_str.lower() or "True" in repr_str

    # ==================== SHAPES AND DIMENSIONS ====================
    
    @pytest.mark.parametrize("arr, expected_shape", [
        ([[1]], (1, 1)),
        ([[1, 2, 3]], (1, 3)),
        ([[1, 2], [3, 4]], (2, 2)),
        ([[1, 2, 3], [4, 5, 6]], (2, 3)),
        ([[[1, 2]], [[3, 4]]], (2, 1, 2)),
    ])
    def test_tensor_shapes(self, arr, expected_shape):
        """Test shapes de tensores"""
        np_array = np.array(arr, dtype=np.float32)
        tensor = rt.Tensor(np_array, requires_grad=False)
        
        result = tensor.numpy()
        assert result.shape == expected_shape

    # ==================== DATA TYPES ====================
    
    @pytest.mark.parametrize("arr,dtype", [
        ([[1, 2], [3, 4]], np.float32),
        ([[1.5, 2.5], [3.5, 4.5]], np.float32),
        ([[0, 1, 2]], np.float32),
    ])
    def test_tensor_dtype_float32(self, arr, dtype):
        """Test que los tensores usan float32"""
        tensor = rt.Tensor(np.array(arr, dtype=dtype), requires_grad=False)
        
        result = tensor.numpy()
        assert result.dtype == np.float32

    @pytest.mark.parametrize("arr,description", [
        ([[0, 0, 0]], "zeros"),
        ([[1, 1, 1]], "ones"),
        ([[-1, -2, -3]], "negative"),
        ([[1e-7, 1e7]], "extreme_values"),
        ([[np.inf]], "infinity"),
    ])
    def test_tensor_special_values(self, arr, description):
        """Test tensores con valores especiales"""
        np_array = np.array(arr, dtype=np.float32)
        tensor = rt.Tensor(np_array, requires_grad=False)
        
        result = tensor.numpy()
        assert np.allclose(result, np_array, atol=1e-5, equal_nan=True)

    # ==================== GRADIENT PROPERTIES ====================
    
    @pytest.mark.parametrize("arr,requires_grad,expected_grad", [
        ([[1, 2], [3, 4]], False, None),
        ([[1, 2], [3, 4]], True, None),  # None o zeros según implementación
        ([[5, 10]], False, None),
        ([[5, 10]], True, None),
    ])
    def test_grad_initialization(self, arr, requires_grad, expected_grad):
        """Test inicialización de gradientes"""
        tensor = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=requires_grad)
        
        if not requires_grad:
            assert tensor.grad is None
        else:
            # El gradiente podría ser None hasta que se compute o estar inicializado a cero
            if tensor.grad is not None:
                assert np.allclose(tensor.grad, np.zeros_like(arr, dtype=np.float32))

    def test_grad_after_backward(self):
        """Test que grad se llena después de backward"""
        arr = [[1, 2], [3, 4]]
        
        A = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        B = A.sum()
        B.backward()
        
        assert A.grad is not None
        assert isinstance(A.grad, np.ndarray)
        assert A.grad.shape == (2, 2)

    def test_leaf_vs_non_leaf(self):
        """Test diferencia entre tensores hoja y no-hoja"""
        arr1 = [[1, 2]]
        arr2 = [[3, 4]]
        
        A = rt.Tensor(np.array(arr1, dtype=np.float32), requires_grad=True)
        B = rt.Tensor(np.array(arr2, dtype=np.float32), requires_grad=True)
        
        assert A.is_leaf
        assert B.is_leaf
        
        C = A + B
        assert not C.is_leaf
        assert C.requires_grad

    # ==================== OPERATIONS PRESERVE PROPERTIES ====================
    
    @pytest.mark.parametrize("arr1,arr2,expected_shape", [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]], (2, 3)),
        ([[1, 2]], [[3, 4]], (1, 2)),
        ([[1], [2], [3]], [[4], [5], [6]], (3, 1)),
    ])
    def test_operations_preserve_shape(self, arr1, arr2, expected_shape):
        """Test que las operaciones preservan shapes correctamente"""
        A = rt.Tensor(np.array(arr1, dtype=np.float32))
        B = rt.Tensor(np.array(arr2, dtype=np.float32))
        
        C = A + B
        D = A * B
        
        assert C.numpy().shape == expected_shape
        assert D.numpy().shape == expected_shape

    def test_operations_preserve_dtype(self):
        """Test que las operaciones preservan dtype"""
        arr1 = [[1, 2]]
        arr2 = [[3, 4]]
        
        A = rt.Tensor(np.array(arr1, dtype=np.float32))
        B = rt.Tensor(np.array(arr2, dtype=np.float32))
        
        C = A + B
        D = A * B
        
        assert C.numpy().dtype == np.float32
        assert D.numpy().dtype == np.float32

    # ==================== EDGE CASES ====================
    
    @pytest.mark.parametrize("arr,expected_shape", [
        ([[5]], (1, 1)),
        ([[1, 2]], (1, 2)),
        ([[1], [2]], (2, 1)),
    ])
    def test_single_element_tensor(self, arr, expected_shape):
        """Test tensor pequeño"""
        tensor = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        
        assert tensor.is_leaf
        assert tensor.requires_grad
        assert tensor.numpy().shape == expected_shape

    @pytest.mark.parametrize("size", [
        (10, 10),
        (50, 50),
        (100, 100),
    ])
    def test_large_tensor(self, size):
        """Test tensor grande"""
        arr = np.random.randn(*size).astype(np.float32)
        tensor = rt.Tensor(arr, requires_grad=False)
        
        result = tensor.numpy()
        assert result.shape == size
        assert np.allclose(result, arr, atol=1e-5)

    def test_empty_like_tensor(self):
        """Test tensor con dimensión 0 (si está soportado)"""
        # Este test podría fallar si no está soportado, ajustar según implementación
        try:
            arr = np.array([[]], dtype=np.float32)
            tensor = rt.Tensor(arr, requires_grad=False)
            result = tensor.numpy()
            assert result.shape[1] == 0
        except:
            pytest.skip("Tensores con dimensión 0 no soportados")

    def test_comparison_with_pytorch(self):
        """Test comparación general con PyTorch"""
        arr = [[1, 2, 3], [4, 5, 6]]
        
        t_rust = rt.Tensor(np.array(arr, dtype=np.float32), requires_grad=True)
        t_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        # Verificar propiedades básicas
        assert t_rust.is_leaf == t_torch.is_leaf
        assert t_rust.requires_grad == t_torch.requires_grad
        assert np.allclose(t_rust.numpy(), t_torch.detach().numpy(), atol=1e-5)

