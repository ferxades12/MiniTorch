import src as M
import torch
import pytest
from src.nn.activations import *
from src.nn.losses import *
from src.nn.optimizers import *
import numpy as np

class TestOptimizers:
    """Test suite completo para optimizadores"""
    
    def setup_method(self):
        """Setup para cada test"""
        np.random.seed(42)
        torch.manual_seed(42)

    # ==================== SGD TESTS ====================
    
    @pytest.mark.parametrize("lr", [0.01, 0.1])
    @pytest.mark.parametrize("momentum", [0.0, 0.9])
    @pytest.mark.parametrize("dampening", [0.0, 0.5])
    @pytest.mark.parametrize("maximize", [False, True])
    def test_sgd_comprehensive(self, lr, momentum, dampening, maximize):
        """Test completo de SGD con todos los parámetros"""
        # Semilla fija para cada test
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Datos simples y determinísticos para evitar variabilidad
        x_np = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]], dtype=np.float32)
        w_np = np.array([[0.1, 0.2],
                         [0.3, 0.4],
                         [0.5, 0.6]], dtype=np.float32)
        b_np = np.array([0.1, 0.2], dtype=np.float32)

        # MiniTorch
        x = M.Tensor(x_np, requires_grad=True)
        w = M.Tensor(w_np, requires_grad=True)
        b = M.Tensor(b_np, requires_grad=True)

        # PyTorch
        tx = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w_np, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)

        # Forward pass
        out = x.dot(w) + b
        loss = out.sum()
        
        tout = tx @ tw + tb
        tloss = tout.sum()

        # Backward pass
        loss.backward()
        tloss.backward()

        # Verificar gradientes antes del paso
        assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-6)
        assert np.allclose(w.grad, tw.grad.numpy(), atol=1e-6)
        assert np.allclose(b.grad, tb.grad.numpy(), atol=1e-6)

        # Optimizadores
        optimizer = SGD([x, w, b], lr=lr, momentum=momentum, dampening=dampening, maximize=maximize)
        torch_optimizer = torch.optim.SGD([tx, tw, tb], lr=lr, momentum=momentum, dampening=dampening, maximize=maximize)

        # Paso de optimización
        optimizer.step()
        torch_optimizer.step()

        # Verificaciones con tolerancia adaptativa
        tolerance = 1e-4 if momentum > 0 else 1e-5
        
        assert np.allclose(x.data, tx.detach().numpy(), atol=tolerance), \
            f"x failed: lr={lr}, momentum={momentum}, dampening={dampening}, maximize={maximize}"
        assert np.allclose(w.data, tw.detach().numpy(), atol=tolerance), \
            f"w failed: lr={lr}, momentum={momentum}, dampening={dampening}, maximize={maximize}"
        assert np.allclose(b.data, tb.detach().numpy(), atol=tolerance), \
            f"b failed: lr={lr}, momentum={momentum}, dampening={dampening}, maximize={maximize}"

    def test_sgd_multiple_steps(self):
        """Test SGD a través de múltiples pasos de optimización"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Datos de prueba
        x_data = np.random.randn(5, 4).astype(np.float32)
        w_data = np.random.randn(4, 3).astype(np.float32)
        
        # MiniTorch
        x = M.Tensor(x_data, requires_grad=True)
        w = M.Tensor(w_data, requires_grad=True)
        
        # PyTorch
        tx = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w_data, dtype=torch.float32, requires_grad=True)
        
        # Optimizadores
        optimizer = SGD([x, w], lr=0.01, momentum=0.9)
        torch_optimizer = torch.optim.SGD([tx, tw], lr=0.01, momentum=0.9)
        
        # Múltiples pasos
        for step in range(5):
            # Zero gradients
            optimizer.zero_grad()
            torch_optimizer.zero_grad()
            
            # Forward
            out = x.dot(w)
            loss = out.sum()
            
            tout = tx @ tw
            tloss = tout.sum()
            
            # Backward
            loss.backward()
            tloss.backward()
            
            # Step
            optimizer.step()
            torch_optimizer.step()
            
            # Verificar que los parámetros siguen siendo similares
            assert np.allclose(x.data, tx.detach().numpy(), atol=1e-4), f"Failed at step {step}"
            assert np.allclose(w.data, tw.detach().numpy(), atol=1e-4), f"Failed at step {step}"

    def test_sgd_edge_cases(self):
        """Test casos edge de SGD"""
        # Caso 1: Gradientes cero
        x = M.Tensor([1.0, 2.0], requires_grad=True)
        x.grad = np.zeros_like(x.data)
        
        optimizer = SGD([x], lr=0.1, momentum=0.9)
        original_data = x.data.copy()
        
        optimizer.step()
        
        # Con gradiente cero, los parámetros no deben cambiar
        assert np.allclose(x.data, original_data)
        
        # Caso 2: Sin gradiente (None)
        y = M.Tensor([3.0, 4.0], requires_grad=True)
        # No asignamos y.grad (queda como None)
        
        optimizer = SGD([y], lr=0.1)
        original_data = y.data.copy()
        
        optimizer.step()  # No debe fallar
        
        # Sin gradiente, no debe cambiar
        assert np.allclose(y.data, original_data)

    # ==================== ADAM TESTS ====================

    @pytest.mark.parametrize("lr", [0.001, 0.01])
    @pytest.mark.parametrize("beta1", [0.9, 0.95])
    @pytest.mark.parametrize("beta2", [0.999, 0.99])
    @pytest.mark.parametrize("eps", [1e-8, 1e-6])
    def test_adam_comprehensive(self, lr, beta1, beta2, eps):
        """Test completo de Adam con todos los parámetros"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Datos simples y determinísticos
        x_np = np.array([[1.0, 2.0],
                         [3.0, 4.0]], dtype=np.float32)
        w_np = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6]], dtype=np.float32)
        
        # MiniTorch
        x = M.Tensor(x_np, requires_grad=True)
        w = M.Tensor(w_np, requires_grad=True)
        
        # PyTorch
        tx = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w_np, dtype=torch.float32, requires_grad=True)
        
        # Forward y backward
        out = x.dot(w)
        loss = out.sum()
        
        tout = tx @ tw
        tloss = tout.sum()
        
        loss.backward()
        tloss.backward()
        
        # Verificar gradientes
        assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-6)
        assert np.allclose(w.grad, tw.grad.numpy(), atol=1e-6)
        
        # Optimizadores
        optimizer = Adam([x, w], lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        torch_optimizer = torch.optim.Adam([tx, tw], lr=lr, betas=(beta1, beta2), eps=eps)
        
        # Paso
        optimizer.step()
        torch_optimizer.step()
        
        # Verificaciones
        tolerance = 1e-4
        assert np.allclose(x.data, tx.detach().numpy(), atol=tolerance), \
            f"x failed: lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}"
        assert np.allclose(w.data, tw.detach().numpy(), atol=tolerance), \
            f"w failed: lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}"

    def test_adam_multiple_steps(self):
        """Test Adam a través de múltiples pasos"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Datos de prueba
        x_data = np.random.randn(3, 4).astype(np.float32)
        w_data = np.random.randn(4, 2).astype(np.float32)
        
        # MiniTorch
        x = M.Tensor(x_data, requires_grad=True)
        w = M.Tensor(w_data, requires_grad=True)
        
        # PyTorch
        tx = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w_data, dtype=torch.float32, requires_grad=True)
        
        # Optimizadores
        optimizer = Adam([x, w], lr=0.001, beta1=0.9, beta2=0.999)
        torch_optimizer = torch.optim.Adam([tx, tw], lr=0.001, betas=(0.9, 0.999))
        
        # Múltiples pasos
        for step in range(10):  # Más pasos para Adam
            # Zero gradients
            optimizer.zero_grad()
            torch_optimizer.zero_grad()
            
            # Forward (función cuadrática simple)
            out = (x.dot(w)).sum()
            loss = out * out  # Función no lineal
            
            tout = (tx @ tw).sum()
            tloss = tout * tout
            
            # Backward
            loss.backward()
            tloss.backward()
            
            # Step
            optimizer.step()
            torch_optimizer.step()
            
            # Verificar convergencia similar
            tolerance = 1e-3 if step < 5 else 1e-4  # Más tolerante al inicio
            assert np.allclose(x.data, tx.detach().numpy(), atol=tolerance), f"Failed at step {step}"
            assert np.allclose(w.data, tw.detach().numpy(), atol=tolerance), f"Failed at step {step}"

    def test_adam_bias_correction(self):
        """Test corrección de sesgo de Adam en los primeros pasos"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Parámetro simple
        x_data = np.array([1.0, 2.0], dtype=np.float32)
        
        x = M.Tensor(x_data, requires_grad=True)
        tx = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
        
        optimizer = Adam([x], lr=0.001)
        torch_optimizer = torch.optim.Adam([tx], lr=0.001)
        
        # Primer paso con gradiente constante
        x.grad = np.array([1.0, 1.0], dtype=np.float32)
        tx.grad = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        optimizer.step()
        torch_optimizer.step()
        
        # En el primer paso, la corrección de sesgo debe ser significativa
        assert np.allclose(x.data, tx.detach().numpy(), atol=1e-6)
        
        # Verificar que los momentos se inicializaron correctamente
        assert hasattr(optimizer, 'm') and len(optimizer.m) > 0
        assert hasattr(optimizer, 'v') and len(optimizer.v) > 0

    def test_adam_edge_cases(self):
        """Test casos edge de Adam"""
        # Caso 1: Gradientes muy pequeños
        x = M.Tensor([1.0, 2.0], requires_grad=True)
        x.grad = np.array([1e-10, 1e-10], dtype=np.float32)
        
        optimizer = Adam([x], lr=0.001)
        original_data = x.data.copy()
        
        optimizer.step()
        
        # Verificar que SÍ hace cambios (porque Adam amplifica gradientes pequeños con bias correction)
        change = np.abs(x.data - original_data)
        print(f"Cambio con gradientes pequeños: {change}")
        
        # Adam debería hacer cambios pequeños pero detectables
        assert np.all(change > 1e-8), "Adam debe hacer algún cambio incluso con gradientes muy pequeños"
        assert np.all(change < 1e-4), "Pero el cambio debe ser razonablemente pequeño"
        
        # Caso 2: Gradientes grandes
        y = M.Tensor([1.0, 2.0], requires_grad=True)
        y.grad = np.array([1000.0, 1000.0], dtype=np.float32)
        
        optimizer = Adam([y], lr=0.001)
        original_data = y.data.copy()
        
        optimizer.step()
        
        # Adam debe normalizar y hacer cambios razonables
        change = np.abs(y.data - original_data)
        print(f"Cambio con gradientes grandes: {change}")
        assert np.all(change < 1.0), "Cambios moderados incluso con gradientes grandes"
        assert np.all(change > 1e-6), "Pero debe hacer cambios detectables"
        
        # Caso 3: Gradientes exactamente cero
        z = M.Tensor([1.0, 2.0], requires_grad=True)
        z.grad = np.array([0.0, 0.0], dtype=np.float32)
        
        optimizer = Adam([z], lr=0.001)
        original_data = z.data.copy()
        
        optimizer.step()
        
        # Con gradientes cero, NO debe cambiar
        assert np.allclose(z.data, original_data, atol=1e-10), "Con gradientes cero no debe haber cambios"

    # ==================== COMPARATIVE TESTS ====================

    def test_optimizers_comparison(self):
        """Test comparativo entre SGD y Adam"""
        np.random.seed(42)
        
        # Problema de optimización simple
        x_data = np.array([[2.0, 3.0]], dtype=np.float32)
        target = np.array([[1.0, 1.0]], dtype=np.float32)
        
        # SGD
        x_sgd = M.Tensor(x_data.copy(), requires_grad=True)
        optimizer_sgd = SGD([x_sgd], lr=0.1)
        
        # Adam
        x_adam = M.Tensor(x_data.copy(), requires_grad=True)
        optimizer_adam = Adam([x_adam], lr=0.1)
        
        # Optimizar hacia el target
        for _ in range(50):
            # SGD
            optimizer_sgd.zero_grad()
            loss_sgd = ((x_sgd - M.Tensor(target))**M.Tensor([[2.0, 2.0]])).sum()
            loss_sgd.backward()
            optimizer_sgd.step()
            
            # Adam
            optimizer_adam.zero_grad()
            loss_adam = ((x_adam - M.Tensor(target))**M.Tensor([[2.0, 2.0]])).sum()
            loss_adam.backward()
            optimizer_adam.step()
        
        # Ambos deben converger hacia el target
        assert np.allclose(x_sgd.data, target, atol=1e-1)
        assert np.allclose(x_adam.data, target, atol=1e-1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
