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
tensor = rs_torch.Tensor(data, requires_grad=True)

print(tensor)
print(f"Is leaf: {tensor.is_leaf}")
print(f"Requires grad: {tensor.requires_grad}")
print(f"Data:\n{tensor.numpy()}")
