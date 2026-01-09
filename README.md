This project consist of implementing Pytorch basic functionality, usin only NumPy (the other libs are for /test module)
The objective is to learn how neural networks work internally

## 🚀 Roadmap

- Tensor implementation and basic operations (sum, mul, transpose, etc.) ✅
- Autograd: manual grad calculation and backward propagation ✅
- Activation functions: ReLU, Sigmoid, Tanh, Softmax ✅
- Loss functions: MSE, Cross-Entropy ✅
- Optimizers: Gradient Descent, Adam ✅
- Neural network layers: Linear, Sequential, RNN ✅
- Regularizations: L1, L2, dropout ✅
- Datasets, Dataloaders, random_split ✅
- Practical examples: xor classification, RNN ✅
- Meta + Kernel architecture (preparation for Rust/CUDA) ✅
- Model serialization 
- ML Algorithms 

## 🚬:

- Meta + kernel design (preparation for Rust) ✅
- CUDA integration (CuPy) 
- Rust backend 
- Rust parallelism 
- CUDA integration in Rust 



## 📂 Project Structure<br>
```
MiniTorch/
│
├── src/
│ ├── ops/                   
│ │ ├── autograd.py          # Operation classes with forward/backward
│ │ ├── dispatch.py          # Dispatch system and meta logic
│ │ ├── cpu.py               # Pure CPU kernels (NumPy)
│ │ └── cuda.py              # GPU kernels
│ ├── nn/
│ │ ├── activations.py       # Activation functions (ReLU, Sigmoid, Tanh, Softmax)
│ │ ├── losses.py            # Loss functions (MSE, CrossEntropy)
│ │ ├── functional.py        # Non-class functions (wrappers)
│ │ ├── layers.py            # Network layers (Linear, Sequential, RNN, Dropout)
│ │ ├── regularizations.py   # L1, L2
│ │ ├── optimizers.py        # SGD, Adam
│ │ └── module.py            # Module base class
│ ├── utils/
│ │ └── data.py              # Dataset, DataLoader, random_split
│ ├── base.py                # Function base class for autograd
│ └── tensor.py              # Tensor class with autograd
├── examples/                 # Usage examples
├── README.md
└── requirements.txt
```


## Installation

Clone the repository:
```bash
git clone https://github.com/ferxades12/MiniTorch
cd MiniTorch
```

Create virtual environment (optional but recommended):
```bash
python -m venv venv
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## GPU Support (CUDA)

If you want to use GPU acceleration with CUDA, you'll need to install CuPy separately along with NVIDIA CUDA Toolkit. CuPy is not included in requirements.txt as it requires specific NVIDIA drivers and CUDA versions. Visit [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for detailed instructions.

```bash
# Example for CUDA 13.x
pip install cupy-cuda13x
```
