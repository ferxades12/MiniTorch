from src.ops.autograd import *
import numpy as np
import cupy as cp

class Tensor:
    grad: np.ndarray
    grad_fn : Function
    is_leaf: bool
    device: str

    def __init__(self, data, requires_grad=False, device="cpu"):
        """Creates a new tensor (numpy array wrapper)

        Args:
            data (array_like): data to be stored in the tensor
            requires_grad (bool, optional): if True, the tensor will track operations for autograd. Defaults to False.
            device (str, optional): Device to store the tensor on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """

        # Data type handling
        if not isinstance(data, (cp.ndarray, np.ndarray)):
            data = np.array(data, dtype=np.float32)

        # Device handling
        if device == "cuda":
            self.data = cp.asarray(data)
        elif device == "cpu":
            if isinstance(data, cp.ndarray):
                self.data = cp.asnumpy(data)
            else:
                self.data = data
        else:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        # Tensor attributes
        self.requires_grad = requires_grad
        self.is_leaf = True

        # Grad initialization
        if self.requires_grad:
            if device == "cuda":
                self.grad = cp.zeros(self.data.shape, dtype=cp.float32)
            else:
                self.grad = np.zeros(self.data.shape, dtype=np.float32)
        else:
            self.grad = None

    def __str__(self):
        return f"Tensor({self.data}, device='{self.device}')"
    def __repr__(self):
        return f"Tensor({self.data}, device='{self.device}')"
    def __getitem__(self, index):
        return self.data[index]
    def __setitem__(self, index, item):
        self.data[index] = item
    def __mul__(self, other):
        return self._apply_binary_op(Mul, other)
    def __rmul__(self, other):
        return self._apply_binary_op(Mul, other, reverse = True)
    def __add__(self, other):
        return self._apply_binary_op(Add, other)
    def __radd__(self, other):
        return self._apply_binary_op(Add, other, reverse=True)
    def __pow__(self, index):
        return self._apply_binary_op(Pow, index)
    def __rpow__(self, base):
        return self._apply_binary_op(Pow, base, reverse=True)
    def __truediv__(self, other):
        return self._apply_binary_op(Div, other)
    def __rtruediv__(self, other):
        return self._apply_binary_op(Div, other, reverse=True)
    def __sub__(self, other):
        return self._apply_binary_op(Sub, other)
    def __rsub__(self, other):
        return self._apply_binary_op(Sub, other, reverse=True)
    def dot(self, other):
        return self._apply_binary_op(Dot, other)
    def T(self):
        return self._apply_unary_op(Transpose)
    def sum(self, axis=None):
        return self._apply_unary_op(Sum, axis)
    def abs(self):
        return self._apply_unary_op(Abs)
    def mean(self):
        return self.sum() / self.numel()
    def log(self):
        return self._apply_unary_op(Log)

    def _apply_binary_op(self, op, other, reverse = False) -> 'Tensor':
        """
        Applies a binary operation between self and other

        Normalises variables to Tensor if necessary
        Instantiates operation class and forwards the data
        Updates grad_fn with that instance if required

        reverse changes order (useful to rmul-like ops)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)

        op = op()

        if reverse:
            result = op.forward(other, self)
        else:
            result = op.forward(self, other)

        if self.requires_grad:
            result.grad_fn = op

        return result
    
    def _apply_unary_op(self, op, *args) -> 'Tensor':
        """
        Applies a unary operation to self

        Instantiates operation class and forwards the data
        Updates grad_fn with that instance if required
        """

        op = op()

        result = op.forward(self, *args)

        if self.requires_grad:
            result.grad_fn = op

        return result

    def backward(self, grad=None):
        """
        Initiates backward operation
        Implicit gradient is only allowed for scalars, otherwise, a direction must be provided
        """
        if self.numel() != 1 and grad is None:
            raise RuntimeError('grad can be implicitly created only for scalars')

        xp = cp if self.device == "cuda" else np
        grad = grad if grad is not None else xp.ones_like(self.data)
        self.grad_fn.backward(grad)

    def shape(self):
        return self.data.shape

    def numel(self) -> int:
        """
        Returns the number of elements in the tensor
        """
        n = 1
        for dim in self.shape():
            n *= dim
        return n
    
    def numdims(self) -> int:
        """
        Returns the number of dimensions of the tensor
        """
        return len(self.shape())
    
    def reshape(self, dims) -> 'Tensor':
        reshaped = Tensor(self.data.reshape(dims), self.requires_grad, device=self.device)

        if self.requires_grad and self.grad is not None:
            reshaped.grad = self.grad.reshape(dims)
        
        return reshaped
    
    def one_hot(self, length) -> np.ndarray:
        if self.numdims() != 1:
            raise ValueError("one_hot es solo para vectores")

        xp = cp if self.device == "cuda" else np
        one_hot = xp.zeros((self.shape()[0], length))
        one_hot[xp.arange(len(self.data)), self.data.astype(int)] = 1

        return one_hot

    def copy(self) -> 'Tensor':
        copy = Tensor(self.data.copy(), self.requires_grad)
        if self.grad is not None:
            copy.grad = self.grad.copy()
        
        if hasattr(self, "grad_fn"):
            copy.grad_fn = self.grad_fn

        copy.is_leaf = self.is_leaf
        return copy
    
    def empty_like(self, requires_grad:bool = False):
        xp = cp if self.device == "cuda" else np
        return Tensor(xp.empty_like(self.data), requires_grad, device=self.device)
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU"""
        if self.device == "cpu":
            return self
        else:
            return Tensor(cp.asnumpy(self.data), self.requires_grad, device="cpu")
    
    def cuda(self) -> 'Tensor':
        if self.device == "cuda":
            return self
        else:
            return Tensor(cp.asarray(self.data), self.requires_grad, device="cuda")
    
    @property # Tratarlo como un atributo
    def device(self):
        return "cuda" if isinstance(self.data, cp.ndarray) else "cpu"
    

def stack(tensors, axis=0):
        tensors = [tensor.data for tensor in tensors]

        return Tensor(np.stack(tensors, axis=axis))

def maximum(A, B) -> 'Tensor':
    """
    returns the maximun bit-wise of A and B
    """
    A = A if isinstance(A, Tensor) else Tensor(A)

    return A._apply_binary_op(Maximum, B)

def minimum(A, B) -> 'Tensor':
    """
    returns the minimum bit-wise of A and B
    """
    A = A if isinstance(A, Tensor) else Tensor(A)

    return A._apply_binary_op(Minimum, B)

def random(*shape, device="cpu") -> 'Tensor':
    """
    returns a tensor with the given shape, filled with random values from a uniform distribution [0, 1)
    """
    xp = cp if device == "cuda" else np
    return Tensor(xp.random.rand(*shape), device=device)

def empty_like(*shape, requires_grad:bool = False, device="cpu") -> 'Tensor':
    xp = cp if device == "cuda" else np
    return Tensor(xp.empty_like(*shape), requires_grad, device=device)