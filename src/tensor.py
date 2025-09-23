from src.operations import *

class Tensor:
    grad = None
    grad_fn = None
    is_leaf = True

    def __init__(self, data, requires_grad=False):
        """
        Creates a numpy array with the data given
        Initializes grad variable if required with the shape of the data
        """

        self.data = np.array(data)
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = np.zeros(self.data.shape)

    def __str__(self):
        return str(self.data)
    def __mul__(self, other):
        return self._apply_binary_op(Mul, other)
    def __rmul__(self, other):
        return self._apply_binary_op(Mul, other, reverse = True)
    def __add__(self, other):
        return self._apply_binary_op(Add, other)
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
        return self._apply_unary_op(Traspose)
    def sum(self):
        return self._apply_unary_op(Sum)
    

    def _apply_binary_op(self, op, other, reverse = False):
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
    
    def _apply_unary_op(self, op):
        """
        Applies a unary operation to self

        Instantiates operation class and forwards the data
        Updates grad_fn with that instance if required
        """

        op = op()

        result = op.forward(self)

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

        grad = grad if grad is not None else 1
        self.grad_fn.backward(grad)

    def shape(self):
        return self.data.shape

    def numel(self):
        """
        Returns the number of elements in the tensor
        """
        n = 1
        for dim in self.shape():
            n *= dim
        return n
    
    def numdims(self):
        """
        Returns the number of dimensions of the tensor
        """
        return len(self.shape())
    
    def reshape(self, dims):
        return self.data.reshape(dims)

def maximum(A, B):
    """
    returns the maximun bit-wise of A and B
    """
    A = A if isinstance(A, Tensor) else Tensor(A)

    return A._apply_binary_op(Maximum, B)

def minimum(A, B):
    """
    returns the minimum bit-wise of A and B
    """
    A = A if isinstance(A, Tensor) else Tensor(A)

    return A._apply_binary_op(Minimum, B)

def sigmoid(tensor):
    """
    Applies the sigmoid function element-wise
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    return tensor._apply_unary_op(Sigmoid)

def softmax(tensor):
    """
    Applies the softmax function to a tensor
    """
    tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
    
    if tensor.numdims() == 1:
        return tensor._apply_unary_op(Softmax1D)
    else:
        raise ValueError("Softmax not implemented for more than 1 dim")

def main():
    # Tensores con gradiente
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([2, 3], requires_grad=True)

    # Producto punto
    a = y.dot(x)  # vector · matriz → vector
    # Multiplicación por un escalar
    b = a * 2
    # Elevar al cuadrado
    c = b**2

    d = c.sum()
    print(c)
    # Backprop
    d.backward()

    # Resultados
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("c:", c.data)


if __name__ == "__main__":
    main()