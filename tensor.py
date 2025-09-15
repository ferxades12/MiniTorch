from functions import *

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
    def dot(self, other):
        return self._apply_op(Dot, other)

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

        self.grad_fn.backward(np.ones(self.shape()))


    def shape(self):
        return self.data.shape


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

    # Backprop
    c.backward()

    # Resultados
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("c:", c.data)


if __name__ == "__main__":
    main()