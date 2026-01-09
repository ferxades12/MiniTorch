import src as M
import pytest
import numpy as np
import cupy as cp

class TestCUDAOperations:
    """Test suite completo para operaciones con tensores en GPU (CUDA)"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        cp.random.seed(42)
        
    def _compare_cpu_cuda(self, cpu_tensor, cuda_tensor, atol=1e-5):
        """Helper para comparar tensores CPU vs CUDA"""
        cuda_data_cpu = cp.asnumpy(cuda_tensor.data)
        assert np.allclose(cpu_tensor.data, cuda_data_cpu, atol=atol), \
            f"CPU and CUDA results differ: max diff = {np.abs(cpu_tensor.data - cuda_data_cpu).max()}"
        
    def _compare_grads(self, cpu_tensor, cuda_tensor, atol=1e-5):
        """Helper para comparar gradientes CPU vs CUDA"""
        if cpu_tensor.grad is not None and cuda_tensor.grad is not None:
            cuda_grad_cpu = cp.asnumpy(cuda_tensor.grad)
            assert np.allclose(cpu_tensor.grad, cuda_grad_cpu, atol=atol), \
                f"CPU and CUDA gradients differ: max diff = {np.abs(cpu_tensor.grad - cuda_grad_cpu).max()}"

    @pytest.mark.parametrize("data, scalar", [
        ([[1, 2, 3], [4, 5, 6]], 2.0),
        ([[1.5, 2.5], [3.5, 4.5]], 0.5),
        ([[-1, 0, 1]], 3),
        ([[0.1, 0.2, 0.3]], 10),
    ])
    def test_add_scalar(self, data, scalar):
        """Test suma tensor + escalar en CPU vs CUDA"""
        cpu_A = M.Tensor(data, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(data, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A + scalar
        cuda_C = cuda_A + scalar
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("a, b", [
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1.5, 2.5, 3.5]], [[0.5, 1.0, 1.5]]),
        ([[-1, 0, 1], [2, -2, 3]], [[1, 1, 1], [-1, -1, -1]]),
    ])
    def test_add(self, a, b):
        """Test suma tensor + tensor en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A + cpu_B
        cuda_C = cuda_A + cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    @pytest.mark.parametrize("a, b", [
        ([[1, 2, 3]], [[4, 5, 6]]),
        ([[2, 4], [6, 8]], [[1, 2], [3, 4]]),
        ([[10, 20, 30]], [[2, 4, 5]]),
    ])
    def test_mul(self, a, b):
        """Test multiplicación elemento a elemento en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A * cpu_B
        cuda_C = cuda_A * cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    @pytest.mark.parametrize("a, b", [
        ([[5, 10, 15]], [[1, 2, 3]]),
        ([[10, 20], [30, 40]], [[2, 4], [5, 8]]),
        ([[100]], [[4]]),
    ])
    def test_div(self, a, b):
        """Test división en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A / cpu_B
        cuda_C = cuda_A / cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    @pytest.mark.parametrize("a, b", [
        ([[10, 5, 8]], [[2, 3, 1]]),
        ([[100, 50]], [[10, 25]]),
    ])
    def test_sub(self, a, b):
        """Test resta en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A - cpu_B
        cuda_C = cuda_A - cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    @pytest.mark.parametrize("base, exp", [
        ([[2, 3, 4]], [[2, 2, 2]]),
        ([[2, 3, 4]], 2),
        ([[1, 2, 3]], [[0.5, 1.5, 2]]),
    ])
    def test_pow(self, base, exp):
        """Test potencia en CPU vs CUDA"""
        cpu_A = M.Tensor(base, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(base, requires_grad=True, device="cuda")
        
        if np.isscalar(exp):
            cpu_C = cpu_A ** exp
            cuda_C = cuda_A ** exp
        else:
            cpu_B = M.Tensor(exp, requires_grad=True, device="cpu")
            cuda_B = M.Tensor(exp, requires_grad=True, device="cuda")
            cpu_C = cpu_A ** cpu_B
            cuda_C = cuda_A ** cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("a, b", [
        ([1, 2, 3], [4, 5, 6]),  # vector dot vector
        ([[1, 2]], [[3], [4]]),  # matriz @ vector
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),  # matriz @ matriz
        ([[1, 2, 3]], [[4], [5], [6]]),  # 1x3 @ 3x1
    ])
    def test_dot(self, a, b):
        """Test producto punto/matricial en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A.dot(cpu_B)
        cuda_C = cuda_A.dot(cuda_B)
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2], [3, 4], [5, 6]],
        [[1.5, 2.5, 3.5]],
    ])
    def test_transpose(self, arr):
        """Test transposición en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        cpu_B = cpu_A.T()
        cuda_B = cuda_A.T()
        
        self._compare_cpu_cuda(cpu_B, cuda_B)
        
        cpu_B.sum().backward()
        cuda_B.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("arr, axis", [
        ([[1, 2, 3], [4, 5, 6]], None),  # sum todo
        ([[1, 2, 3], [4, 5, 6]], 0),  # sum por columnas
        ([[1, 2, 3], [4, 5, 6]], 1),  # sum por filas
        ([[1.5, 2.5], [3.5, 4.5]], None),
    ])
    def test_sum(self, arr, axis):
        """Test suma de elementos en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        cpu_B = cpu_A.sum(axis=axis)
        cuda_B = cuda_A.sum(axis=axis)
        
        self._compare_cpu_cuda(cpu_B, cuda_B)
        
        cpu_B.sum().backward()
        cuda_B.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("arr", [
        [[-1, 2, -3]],
        [[1.5, -2.5, 3.5]],
        [[-10, 0, 10], [5, -5, 0]],
    ])
    def test_abs(self, arr):
        """Test valor absoluto en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        cpu_B = cpu_A.abs()
        cuda_B = cuda_A.abs()
        
        self._compare_cpu_cuda(cpu_B, cuda_B)
        
        cpu_B.sum().backward()
        cuda_B.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[0.1, 0.5, 1.0]],
        [[np.e, np.e**2, np.e**3]],
    ])
    def test_log(self, arr):
        """Test logaritmo natural en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        cpu_B = cpu_A.log()
        cuda_B = cuda_A.log()
        
        self._compare_cpu_cuda(cpu_B, cuda_B)
        
        cpu_B.sum().backward()
        cuda_B.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)

    @pytest.mark.parametrize("a, b", [
        ([[1, 5, 3]], [[4, 2, 6]]),
        ([[1, 2], [3, 4]], [[2, 1], [4, 3]]),
    ])
    def test_maximum(self, a, b):
        """Test máximo elemento a elemento en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        # Usando la función Maximum de autograd
        from src.ops.autograd import Maximum
        
        cpu_max_op = Maximum()
        cuda_max_op = Maximum()
        
        cpu_C = cpu_max_op.forward(cpu_A, cpu_B)
        cuda_C = cuda_max_op.forward(cuda_A, cuda_B)
        
        self._compare_cpu_cuda(cpu_C, cuda_C)

    @pytest.mark.parametrize("a, b", [
        ([[1, 5, 3]], [[4, 2, 6]]),
        ([[1, 2], [3, 4]], [[2, 1], [4, 3]]),
    ])
    def test_minimum(self, a, b):
        """Test mínimo elemento a elemento en CPU vs CUDA"""
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        # Usando la función Minimum de autograd
        from src.ops.autograd import Minimum
        
        cpu_min_op = Minimum()
        cuda_min_op = Minimum()
        
        cpu_C = cpu_min_op.forward(cpu_A, cpu_B)
        cuda_C = cuda_min_op.forward(cuda_A, cuda_B)
        
        self._compare_cpu_cuda(cpu_C, cuda_C)

    @pytest.mark.parametrize("arr", [
        [[-2, 0, 2]],
        [[0, 1, 2, 3]],
        [[-5, -1, 0, 1, 5]],
    ])
    def test_sigmoid(self, arr):
        """Test sigmoid en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        # Usando la función SigmoidOp de autograd
        from src.ops.autograd import SigmoidOp
        
        cpu_sig_op = SigmoidOp()
        cuda_sig_op = SigmoidOp()
        
        cpu_B = cpu_sig_op.forward(cpu_A)
        cuda_B = cuda_sig_op.forward(cuda_A)
        
        self._compare_cpu_cuda(cpu_B, cuda_B, atol=1e-4)

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3]],
        [[0.5, 1.0, 1.5, 2.0]],
        [[1, 2, 3], [4, 5, 6]],  # batch de 2
    ])
    def test_softmax(self, arr):
        """Test softmax en CPU vs CUDA"""
        cpu_A = M.Tensor(arr, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(arr, requires_grad=True, device="cuda")
        
        # Usando la función SoftmaxOp de autograd
        from src.ops.autograd import SoftmaxOp
        
        cpu_soft_op = SoftmaxOp()
        cuda_soft_op = SoftmaxOp()
        
        cpu_B = cpu_soft_op.forward(cpu_A)
        cuda_B = cuda_soft_op.forward(cuda_A)
        
        self._compare_cpu_cuda(cpu_B, cuda_B, atol=1e-4)

    def test_device_transfer(self):
        """Test transferencia de tensores entre CPU y CUDA"""
        # CPU -> CUDA
        cpu_A = M.Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, device="cpu")
        assert cpu_A.device == "cpu"
        
        cuda_A = cpu_A.cuda()
        assert cuda_A.device == "cuda"
        assert isinstance(cuda_A.data, cp.ndarray)
        
        # CUDA -> CPU
        cpu_A2 = cuda_A.cpu()
        assert cpu_A2.device == "cpu"
        assert isinstance(cpu_A2.data, np.ndarray)
        
        # Comparar datos
        assert np.allclose(cpu_A.data, cpu_A2.data)

    def test_large_matrix_operations(self):
        """Test con matrices grandes para verificar rendimiento en GPU"""
        size = 1000
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        cpu_A = M.Tensor(a, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(b, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(a, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(b, requires_grad=True, device="cuda")
        
        # Operación de suma
        cpu_C = cpu_A + cpu_B
        cuda_C = cuda_A + cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C, atol=1e-4)

    def test_backward_propagation(self):
        """Test completo de backward propagation en GPU"""
        data = [[1, 2, 3], [4, 5, 6]]
        
        cpu_A = M.Tensor(data, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(data, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(data, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(data, requires_grad=True, device="cuda")
        
        # Forward
        cpu_C = cpu_A * cpu_B + cpu_A
        cuda_C = cuda_A * cuda_B + cuda_A
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        # Backward
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    def test_broadcasting(self):
        """Test de broadcasting en GPU"""
        # Matriz + vector (broadcasting)
        mat = [[1, 2, 3], [4, 5, 6]]
        vec = [10, 20, 30]
        
        cpu_A = M.Tensor(mat, requires_grad=True, device="cpu")
        cpu_B = M.Tensor(vec, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(mat, requires_grad=True, device="cuda")
        cuda_B = M.Tensor(vec, requires_grad=True, device="cuda")
        
        cpu_C = cpu_A + cpu_B
        cuda_C = cuda_A + cuda_B
        
        self._compare_cpu_cuda(cpu_C, cuda_C)
        
        cpu_C.sum().backward()
        cuda_C.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
        self._compare_grads(cpu_B, cuda_B)

    def test_chain_operations(self):
        """Test de cadena de operaciones complejas en GPU"""
        data = [[1, 2, 3], [4, 5, 6]]
        
        cpu_A = M.Tensor(data, requires_grad=True, device="cpu")
        cuda_A = M.Tensor(data, requires_grad=True, device="cuda")
        
        # Cadena de operaciones: (A * 2 + 3) ** 2 - 1
        cpu_result = ((cpu_A * 2 + 3) ** 2) - 1
        cuda_result = ((cuda_A * 2 + 3) ** 2) - 1
        
        self._compare_cpu_cuda(cpu_result, cuda_result)
        
        cpu_result.sum().backward()
        cuda_result.sum().backward()
        
        self._compare_grads(cpu_A, cuda_A)
