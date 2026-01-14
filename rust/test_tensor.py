"""
Script de prueba para el módulo rs_torch

Para compilar e instalar:
    pip install maturin
    maturin develop --release

Para solo compilar:
    cargo build --release
"""

import numpy as np
import rs_torch

# Crear un tensor desde numpy
data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
A = rs_torch.Tensor(data, requires_grad=True)
B = rs_torch.Tensor(data * 2, requires_grad=True)
C = A*B
print(A + B)
print(f"A - Is leaf: {A.is_leaf}, Requires grad: {A.requires_grad}")
print(f"C - Is leaf: {C.is_leaf}, Requires grad: {C.requires_grad}")
print(f"C Data:\n{C.numpy()}")
D = C.sum()
print(f"D - Is leaf: {D.is_leaf}, Requires grad: {D.requires_grad}")
print(f"D Data: {D.numpy()}")
D.backward()

# Ahora puedes acceder al gradiente directamente como numpy array
print(f"Gradient de C:\n{C.grad}")
print(f"Gradient de A:\n{A.grad}")
print(f"Gradient de B:\n{B.grad}")

# También puedes usarlo directamente sin .numpy()
if C.grad is not None:
    print(f"\nGradiente de C es un numpy array: {type(C.grad)}")
    print(f"Shape: {C.grad.shape}")
