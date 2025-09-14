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
        """
        Normalises variables to Tensor if necessary
        Instantiates operation class and forwards the data
        Updates grad_fn if required with that instance
        """
        other = other if isinstance(other, Tensor) else Tensor(other)

        op = Mul()
        result = op.forward(self, other)

        if self.requires_grad:
            result.grad_fn = op

        return result

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        op = Mul()
        result = op.forward(other, self)

        if self.requires_grad:
            result.grad_fn = op

        return result

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        op = Add()
        result = op.forward(self, other)

        if self.requires_grad:
            result.grad_fn = op

        return result

    def __pow__(self, index):
        index = index if isinstance(index, Tensor) else Tensor(index)

        op = Pow()
        result = op.forward(self, index)

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

class Mul:
    ctx = None

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
        Retrieves the data in ctx

        If requires_grad, checks is_leaf
         - If it is True the grad needs to be accumulated in the node
         - If it is False the grad needs to propagated to the node's father

        ax d/dx = a
        """

        tensor = self.ctx[0]
        other = self.ctx[1]

        grad_tensor = other.data * grad_output
        grad_other = tensor.data * grad_output

        if tensor.requires_grad:
            if tensor.is_leaf:
                tensor.grad += grad_tensor
            else:
                tensor.grad_fn.backward(grad_tensor)

        if other.requires_grad:
            if other.is_leaf:
                other.grad += grad_other
            else:
                other.grad_fn.backward(grad_other)
class Add:
    ctx = None
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
        Retrieves the data in ctx

        If requires_grad, checks is_leaf
         - If it is True the grad needs to be accumulated in the node
         - If it is False the grad needs to propagated to the node's father

        (a + x) d/dx = 1
        """

        tensor = self.ctx[0]
        other = self.ctx[1]

        if tensor.requires_grad:
            if tensor.is_leaf:
                tensor.grad += grad_output
            else:
                tensor.grad_fn.backward(grad_output)

        if other.requires_grad:
            if other.is_leaf:
                other.grad += grad_output
            else:
                other.grad_fn.backward(grad_output)

class Pow:
    ctx = None
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
        Retrieves the data in ctx

        If requires_grad, checks is_leaf
         - If it is True the grad needs to be accumulated in the node
         - If it is False the grad needs to propagated to the node's father

        x^a d/dx = a * x^(a-1)
        a^x d/dx = a^x * ln(a)
        """

        tensor = self.ctx[0]
        index = self.ctx[1]
        result = self.ctx[2]

        tensor_grad = index.data * (np.pow(tensor.data, (index.data - 1))) * grad_output
        index_grad = result * np.log(tensor.data) * grad_output

        if tensor.requires_grad:
            if tensor.is_leaf:
                tensor.grad += tensor_grad
            else:
                tensor.grad_fn.backward(tensor_grad)

        if index.requires_grad:
            if index.is_leaf:
                index.grad += index_grad
            else:
                index.grad_fn.backward(index_grad)


def main():
    # --- TEST COMPLETO ---
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = Tensor(4.0, requires_grad=True)

    # 1️⃣ Multiplicación
    a = x * y  # a = x * y = 2 * 3 = 6

    # 2️⃣ Suma
    b = a + z  # b = a + z = 6 + 4 = 10

    # 3️⃣ Potencia
    c = b ** a  # c = b^a = 10^6 = 1_000_000

    # Retropropagación
    c.backward()

    print("x.grad:", x.grad)  # ≈ 8_707_755
    print("y.grad:", y.grad)  # ≈ 5_805_170
    print("z.grad:", z.grad)  # = 600_000
    print("c:", c.data)  # 1_000_000


if __name__ == "__main__":
    main()