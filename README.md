This project consist of implementing Pytorch basic functionality, usin only NumPy (the other libs are for /test module)
The objective is to learn how neural networks work internally

🚀 Roadmap

Tensor implementation and basic operations (sum, mul, transpose, etc.) ✅

Autograd: manual grad calculation and backward propagation ✅

Activation functions: ReLU, Sigmoid, Tanh, Softmax. ✅

Loss functions: MSE, Cross-Entropy.✅

Optimizers: Gradient Descent, Adam.✅

Neural network layers: Linear, Sequential ✅

Regularizations: L1, L2, dropout ✅

Datasets, Dataloaders, random_split ✅

Model serialization

ML Algorithms

Practical examples: xor classification ✅

🚬:
Diseño meta + kernel (preparación para Rust)

Integración de CUDA (CuPy)

Backend en Rust

Paralelismo en Rust

Integración de CUDA en Rust


📂 Estructura del Proyecto<br>
```
MiniTorch/
│
├── src/
│ ├── nn/
│ │ ├── activations.py # Activation functions
│ │ ├── losses.py # Loss functions
│ │ ├── functional.py # Non-class functions
│ │ ├── layers.py #nn layers
│ │ ├── regularizations.py
│ │ └── optimizers.py # optimizers
│ ├── utils/
│ │ ├── data.py # Dataloaders, etc.
│ ├── tensor.py
│ └── operations.py
├── examples/ # Ejemplos de uso
│ └── xor_classification.py
├── tests/
├── README.md
└── requirements.txt
```


Cloning the repo:
```
git clone <https://github.com/ferxades12/MiniTorch>
cd <MiniTorch>
```
Creating venv (optional but recommended):

```
python -m venv venv
# En Linux/macOS
source venv/bin/activate
# En Windows
venv\Scripts\activate
```

Installing dependencies:
```
pip install -r requirements.txt
```
