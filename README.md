Este proyecto consiste en reescribir las funciones básicas de PyTorch desde cero, utilizando únicamente NumPy. El objetivo es aprender cómo funcionan internamente los tensores, la propagación, y el autograd, entre otros.

🚀 Características

Implementación de tensores y operaciones básicas (suma, multiplicación, transposición, etc.) ✅

Autograd: cálculo manual del gradiente y backward propagation ✅

Funciones de activación: ReLU, Sigmoid, Tanh, Softmax. ✅

Funciones de pérdida: MSE, Cross-Entropy.✅

Optimización con Gradient Descent, Adam.✅

Capas de Redes neuronales: Linear

Regularizacion: L1, L2, dropout 

Model serialization

Algoritmos de Machine Learning

Ejemplos prácticos: regresión lineal, clasificación simple.

📂 Estructura del Proyecto<br>
```
MiniTorch/
│
├── src/
│ ├── nn/
│ │ ├── activations.py # Funciones de activación
│ │ ├── losses.py # Funciones de pérdida
│ │ └── optimizers.py # Optimizadores (SGD)
│ ├── tensor.py # Clase Tensor personalizada
│ └── operations.py # Operaciones básicas (suma, multiplicación, etc.)
├── examples/ # Ejemplos de uso
│ └── linear_regression.py
├── tests/
├── README.md
└── requirements.txt
```
📝 Ejemplo de Uso


📌 Como instalarlo

Clonar el repositorio:
```
git clone <https://github.com/ferxades12/MiniTorch>
cd <MiniTorch>
```
Crear un entorno virtual (opcional pero recomendado):

```
python -m venv venv
# En Linux/macOS
source venv/bin/activate
# En Windows
venv\Scripts\activate
```

Instalar las dependencias:
```
pip install -r requirements.txt
```
