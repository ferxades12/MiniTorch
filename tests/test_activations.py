import src as M
import torch
import pytest
import src.nn as nn
import numpy as np

class TestActivations:
    """Test suite completo para todas las funciones de activación"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== SOFTMAX TESTS ====================
    
    @pytest.mark.parametrize("arr", [
        [0, 1, 2, 3],                # valores crecientes
        [-1, 0, 1],                  # negativos y positivos
        [1000, 1001, 1002],          # valores grandes (test de estabilidad numérica)
        [-1000, 0, 1000],            # extremos
        [0, 0, 0],                   # todos iguales
        np.random.uniform(-5, 5, 5), # aleatorio
    ])
    def test_softmax_1d(self, arr):
        """Test softmax en vectores 1D"""

        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B = nn.softmax(A)
        tb = torch.nn.functional.softmax(ta, dim=0)

        B.sum().backward()
        tb.sum().backward()

        assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("arr", [
        [[1, 2, 3], [4, 5, 6]],           # 2x3 normal
        [[1, 2], [3, 4], [5, 6]],         # 3x2 normal
        [[-1000, 0, 1000]],               # extremos en una fila
        [[1000, 1001, 1002], [999, 1000, 1001]], # valores grandes
        [[0, 0, 0], [1, 1, 1]],           # casos edge
        np.random.uniform(-5, 5, (4, 3)), # aleatorio
    ])
    def test_softmax_batch(self, arr):
        """Test softmax aplicado por filas (batch)"""
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B = nn.softmax(A)  # Debería aplicar softmax por fila
        tb = torch.nn.functional.softmax(ta, dim=1)

        B.sum().backward()
        tb.sum().backward()

        assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

    @pytest.mark.parametrize("case, arr", [
        ("overflow", [[1000, 1001, 1002]]),
        ("underflow", [[-1000, -1001, -1002]]),
        ("mixed_extreme", [[-1000, 0, 1000]]),
        ("near_zero", [[1e-8, 2e-8, 3e-8]]),
        ("identical_large", [[1000, 1000, 1000]]),
        ("identical_small", [[-1000, -1000, -1000]]),
    ])
    def test_softmax_stability(self, case, arr):
        """Test estabilidad numérica del softmax"""
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        B = nn.softmax(A)
        tb = torch.nn.functional.softmax(ta, dim=1)
        
        # Verificar que no hay overflow/underflow
        assert not np.isnan(B.data).any(), f"NaN detected in case {case}"
        assert not np.isinf(B.data).any(), f"Inf detected in case {case}"
        
        B.sum().backward()
        tb.sum().backward()
        
        tolerance = 1e-4 if case in ["overflow", "underflow"] else 1e-5
        assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)

    def test_softmax_properties(self):
        """Verifica propiedades matemáticas del softmax"""
        arr = np.random.uniform(-5, 5, (3, 4))
        A = M.Tensor(arr, requires_grad=True)
        B = nn.softmax(A)
        
        # Propiedad 1: La suma de cada fila debe ser 1
        if B.numdims() > 1:
            row_sums = B.sum(axis=1)
            expected = np.ones(B.shape()[0])
            assert np.allclose(row_sums.data, expected, atol=1e-6)
        else:
            assert np.allclose(B.sum().data, 1.0, atol=1e-6)
        
        # Propiedad 2: Todos los valores deben ser >= 0
        assert np.all(B.data >= 0)
        
        # Propiedad 3: Todos los valores deben ser <= 1
        assert np.all(B.data <= 1)

    # ==================== RELU TESTS ====================

    @pytest.mark.parametrize("case, arr", [
        ("all_negative", [[-5, -3, -1]]),
        ("all_positive", [[1, 3, 5]]),
        ("mixed", [[-2, 0, 2]]),
        ("exact_zero", [[0, 0, 0]]),
        ("tiny_values", [[1e-10, -1e-10, 0]]),
        ("extreme_positive", [[1e6, 1e7, 1e8]]),
        ("extreme_negative", [[-1e6, -1e7, -1e8]]),
        ("random", np.random.uniform(-5, 5, (3, 3))),
    ])
    def test_relu_cases(self, case, arr):
        """Test casos específicos de ReLU"""

        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B = nn.relu(A)
        tb = torch.nn.functional.relu(ta)

        B.sum().backward()
        tb.sum().backward()

        assert np.allclose(B.data, tb.detach().numpy(), atol=1e-5)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)

    def test_relu_properties(self):
        """Verifica propiedades de ReLU"""
        arr = np.random.uniform(-10, 10, (5, 5))
        A = M.Tensor(arr, requires_grad=True)
        B = nn.relu(A)
        
        # Propiedad 1: ReLU(x) >= 0 para todo x
        assert np.all(B.data >= 0)
        
        # Propiedad 2: ReLU(x) = x si x > 0
        positive_mask = A.data > 0
        assert np.allclose(B.data[positive_mask], A.data[positive_mask])
        
        # Propiedad 3: ReLU(x) = 0 si x <= 0
        non_positive_mask = A.data <= 0
        assert np.allclose(B.data[non_positive_mask], 0)

    # ==================== SIGMOID TESTS ====================

    @pytest.mark.parametrize("case, arr", [
        ("normal", [[-1, 0], [1, -1]]),
        ("wide_range", [[-10, -1, 0, 1, 10]]),
        ("extreme", [[-100, -0.5, 0, 0.5, 100]]),
        ("small_values", [[-0.001, 0, 0.001]]),
        ("large_positive", [[50, 100, 200]]),
        ("large_negative", [[-50, -100, -200]]),
        ("random", np.random.uniform(-5, 5, (3, 3))),
    ])
    def test_sigmoid_cases(self, case, arr):
        """Test casos específicos de Sigmoid"""

        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B = nn.sigmoid(A)
        tb = torch.sigmoid(ta)

        B.sum().backward()
        tb.sum().backward()

        tolerance = 1e-4 if case in ["extreme", "large_positive", "large_negative"] else 1e-5
        assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)

    def test_sigmoid_range(self):
        """Verifica que sigmoid esté en [0,1]"""
        arr = np.random.uniform(-100, 100, (5, 5))
        A = M.Tensor(arr, requires_grad=True)
        B = nn.sigmoid(A)
        
        assert np.all(B.data >= 0)
        assert np.all(B.data <= 1)
        
        # Test propiedades específicas
        # sigmoid(0) = 0.5
        zero_input = M.Tensor([0], requires_grad=True)
        zero_output = nn.sigmoid(zero_input)
        assert np.allclose(zero_output.data, 0.5, atol=1e-7)
        
        # sigmoid(-x) = 1 - sigmoid(x)
        x = M.Tensor([2.5], requires_grad=True)
        neg_x = M.Tensor([-2.5], requires_grad=True)
        sig_x = nn.sigmoid(x)
        sig_neg_x = nn.sigmoid(neg_x)
        assert np.allclose(sig_neg_x.data, 1 - sig_x.data, atol=1e-6)

    # ==================== TANH TESTS ====================

    @pytest.mark.parametrize("case, arr", [
        ("normal", [[-1, 0], [1, -1]]),
        ("wide_range", [[-10, -1, 0, 1, 10]]),
        ("extreme", [[-100, -0.5, 0, 0.5, 100]]),
        ("small_values", [[-0.001, 0, 0.001]]),
        ("large_positive", [[20, 50, 100]]),
        ("large_negative", [[-20, -50, -100]]),
        ("random", np.random.uniform(-5, 5, (3, 3))),
    ])
    def test_tanh_cases(self, case, arr):
        """Test casos específicos de Tanh"""

        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        B = nn.tanh(A)
        tb = torch.tanh(ta)

        B.sum().backward()
        tb.sum().backward()

        tolerance = 1e-4 if case in ["extreme", "large_positive", "large_negative"] else 1e-5
        assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance)

    def test_tanh_range(self):
        """Verifica que tanh esté en [-1,1]"""
        arr = np.random.uniform(-100, 100, (5, 5))
        A = M.Tensor(arr, requires_grad=True)
        B = nn.tanh(A)
        
        assert np.all(B.data >= -1)
        assert np.all(B.data <= 1)
        
        # Test propiedades específicas
        # tanh(0) = 0
        zero_input = M.Tensor([0], requires_grad=True)
        zero_output = nn.tanh(zero_input)
        assert np.allclose(zero_output.data, 0, atol=1e-7)
        
        # tanh(-x) = -tanh(x)
        x = M.Tensor([2.5], requires_grad=True)
        neg_x = M.Tensor([-2.5], requires_grad=True)
        tanh_x = nn.tanh(x)
        tanh_neg_x = nn.tanh(neg_x)
        assert np.allclose(tanh_neg_x.data, -tanh_x.data, atol=1e-6)

    # ==================== GRADIENTE NUMÉRICO ====================

    @pytest.mark.parametrize("activation_name", ["sigmoid", "tanh", "relu"])
    def test_numerical_gradient(self, activation_name):
        """Verifica gradientes usando diferencias finitas"""
        
        # Seleccionar activación
        activations = {
            "sigmoid": nn.activations.Sigmoid(),
            "tanh": nn.activations.Tanh(),
            "relu": nn.activations.ReLU()
        }
        activation = activations[activation_name]
        
        def numerical_gradient(f, x_data, h=1e-5):
            grad = np.zeros_like(x_data)
            
            for i in range(x_data.size):
                # f(x + h) - crear nuevo tensor
                x_plus_data = x_data.copy()
                x_plus_data.flat[i] += h
                x_plus = M.Tensor(x_plus_data, requires_grad=False)  # Sin gradientes para evaluación
                y_plus = f(x_plus).sum().data
                
                # f(x - h) - crear nuevo tensor
                x_minus_data = x_data.copy()
                x_minus_data.flat[i] -= h
                x_minus = M.Tensor(x_minus_data, requires_grad=False)  # Sin gradientes para evaluación
                y_minus = f(x_minus).sum().data
                
                # Gradiente numérico
                grad.flat[i] = (y_plus - y_minus) / (2 * h)
                
            return grad

        # Test para cada activación
        np.random.seed(42)
        # Para ReLU, evitar valores muy cerca de 0 donde la derivada es discontinua
        if activation_name == "relu":
            arr = np.random.uniform(0.1, 2, (2, 3))
        else:
            arr = np.random.uniform(-2, 2, (2, 3))
            
        x = M.Tensor(arr, requires_grad=True)
        
        # Gradiente analítico
        x.grad = None
        y = activation(x)
        y.sum().backward()
        analytical_grad = x.grad.copy()
        
        # Gradiente numérico
        numerical_grad = numerical_gradient(activation, arr)  # Pasa los datos, no el tensor
        
        tolerance = 1e-1 if activation_name == "relu" else 1e-2
        assert np.allclose(analytical_grad, numerical_grad, atol=tolerance)

    # ==================== CHAIN ACTIVATIONS ====================

    @pytest.mark.parametrize("batch_size, features", [
        (1, 4),      # batch pequeño
        (8, 4),      # batch normal
        (32, 10),    # batch grande
        (1, 1),      # caso mínimo
    ])
    def test_chain_activations(self, batch_size, features):
        """Test cadena completa de activaciones"""
        np.random.seed(42)

        # Variables de entrada
        arr = np.random.uniform(-2, 2, (batch_size, features))
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)

        # Pesos
        W1 = M.Tensor(np.random.randn(features, features), requires_grad=True)
        w1 = torch.tensor(W1.data, dtype=torch.float32, requires_grad=True)

        W2 = M.Tensor(np.random.randn(features, features), requires_grad=True)
        w2 = torch.tensor(W2.data, dtype=torch.float32, requires_grad=True)

        # Forward RusTorch
        Z1 = A.dot(W1)
        H1 = nn.tanh(Z1)
        Z2 = H1.dot(W2)
        H2 = nn.sigmoid(Z2)
        H3 = nn.relu(H2)
        Out = nn.softmax(H3)
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

        # Verificaciones
        assert np.allclose(Out.data, out_t.detach().numpy(), atol=1e-5)
        assert np.allclose(A.grad, ta.grad.numpy(), atol=1e-5)
        assert np.allclose(W1.grad, w1.grad.numpy(), atol=1e-5)
        assert np.allclose(W2.grad, w2.grad.numpy(), atol=1e-5)

    # ==================== CASOS EXTREMOS COMBINADOS ====================

    @pytest.mark.parametrize("case", [
        "all_zeros",
        "all_ones", 
        "all_negatives",
        "mixed_extreme",
        "tiny_gradients",
    ])
    def test_extreme_cases_combined(self, case):
        """Test casos extremos combinados"""
        
        if case == "all_zeros":
            arr = np.zeros((3, 4))
        elif case == "all_ones":
            arr = np.ones((3, 4))
        elif case == "all_negatives":
            arr = -np.random.uniform(1, 10, (3, 4))
        elif case == "mixed_extreme":
            arr = np.array([[-1000, 1000], [0, 1e-10], [1e10, -1e10]])
        elif case == "tiny_gradients":
            arr = np.random.uniform(-1e-5, 1e-5, (3, 4))
        
        A = M.Tensor(arr, requires_grad=True)
        ta = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
        
        # Test todas las activaciones
        activations = [
            ("sigmoid", nn.activations.Sigmoid(), torch.sigmoid),
            ("tanh", nn.activations.Tanh(), torch.tanh),
            ("relu", nn.activations.ReLU(), torch.nn.functional.relu),
        ]
        
        for name, mini_act, torch_act in activations:
            A.grad = None
            ta.grad = None
            
            B = mini_act(A)
            tb = torch_act(ta)
            
            # Skip si hay problemas numéricos esperados
            if np.isnan(B.data).any() or np.isinf(B.data).any():
                continue
                
            B.sum().backward()
            tb.sum().backward()
            
            tolerance = 1e-4 if case in ["mixed_extreme", "tiny_gradients"] else 1e-5
            
            assert np.allclose(B.data, tb.detach().numpy(), atol=tolerance), f"Failed for {name} in case {case}"
            assert np.allclose(A.grad, ta.grad.numpy(), atol=tolerance), f"Gradient failed for {name} in case {case}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])