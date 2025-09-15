PyTorch From Scratch with NumPy 🐍➕🔢

Este proyecto consiste en reescribir las funciones básicas de PyTorch desde cero, utilizando únicamente NumPy. El objetivo es aprender cómo funcionan internamente los tensores, la propagación, y la autograd.

🚀 Características

Implementación de tensores y operaciones básicas (suma, multiplicación, transposición, etc.).

Funciones de activación: ReLU, Sigmoid, Tanh.

Funciones de pérdida: MSE, Cross-Entropy.

Autograd: cálculo manual del gradiente y backward propagation.

Optimización con Gradient Descent, Adam, RMSProp.

Ejemplos prácticos: regresión lineal, clasificación simple.

📂 Estructura del Proyecto<br>
MiniTorch/<br>
│<br>
├── tensor.py           # Clase Tensor personalizada  <br>
├── operations.py       # Operaciones básicas (suma, multiplicación, etc.)  <br>
├── activations.py      # Funciones de activación<br>
├── losses.py           # Funciones de pérdida<br>
├── optimizers.py       # Optimización (SGD)<br>
├── examples/           # Ejemplos de uso<br>
│   └── linear_regression.py<br>
├── tests/    <br>
└── README.md<br>


📝 Ejemplo de Uso


💡 Motivación


📌 Próximos pasos
Soporte para más funciones de activación y pérdidas.

Implementación de redes neuronales completas (MLP, CNN básicas).
