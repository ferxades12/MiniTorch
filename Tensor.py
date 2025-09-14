import numpy as np


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
        return self._apply_op(Mul, other)
    def __rmul__(self, other):
        return self._apply_op(Mul, other, reverse = True)
    def __add__(self, other):
        return self._apply_op(Add, other)
    def __pow__(self, index):
        return self._apply_op(Pow, index)

    def _apply_op(self, op, other, reverse = False):
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

    def backward(self):
        """
        Initiates backward operation
        """

        self.grad_fn.backward(1)


    def shape(self):
        return self.data.shape


class Function:
    ctx = None

    def _update_grad(self, tensor, grad):
        """
        If requires_grad, checks is_leaf
         - If it is True the grad needs to be accumulated in the node
         - If it is False the grad needs to propagated to the node's father
        """
        if tensor.requires_grad:
            if tensor.is_leaf:
                tensor.grad += grad
            else:
                tensor.grad_fn.backward(grad)


class Mul(Function):
    def forward(self, tensor, other):
        """
        Creates a Tensor result
        Saves necessary data in ctx
        """

        result = Tensor(tensor.data * other.data, requires_grad= tensor.requires_grad or other.requires_grad)
        result.is_leaf = False

        self.ctx = (tensor, other)

        return result

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        ax d/dx = a
        """

        tensor, other = self.ctx

        self._update_grad(tensor, other.data * grad_output)
        self._update_grad(other, tensor.data * grad_output)

class Add(Function):
    def forward(self, tensor, other):
        """
        Creates a Tensor result
        Saves necessary data in ctx
        """

        result = Tensor(tensor.data + other.data, requires_grad=tensor.requires_grad or other.requires_grad)
        result.is_leaf = False

        self.ctx = (tensor, other)

        return result

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (a + x) d/dx = 1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, grad_output)

class Pow(Function):
    def forward(self, tensor, index):
        """
        Creates a Tensor result
        Saves necessary data in ctx
        """

        result = Tensor(np.pow(tensor.data, index.data), requires_grad=tensor.requires_grad or index.requires_grad)
        result.is_leaf = False

        self.ctx = (tensor, index, result.data)

        return result

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        x^a d/dx = a * x^(a-1)
        a^x d/dx = a^x * ln(a)
        """

        tensor, index, result = self.ctx

        tensor_grad = index.data * (np.pow(tensor.data, (index.data - 1))) * grad_output
        index_grad = result * np.log(tensor.data) * grad_output

        self._update_grad(tensor, tensor_grad)
        self._update_grad(index, index_grad)


def main():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = Tensor(4.0, requires_grad=True)

    a = x * y
    b = a + z
    c = b ** a

    c.backward()

    print("x.grad:", x.grad)  # ≈ 8_707_755
    print("y.grad:", y.grad)  # ≈ 5_805_170
    print("z.grad:", z.grad)  # = 600_000
    print("c:", c.data)  # 1_000_000


if __name__ == "__main__":
    main()