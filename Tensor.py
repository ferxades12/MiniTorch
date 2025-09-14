import numpy as np

class Tensor:
    grad = None
    grad_fn = None
    is_leaf = True

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = np.zeros(self.data.shape)

    def __str__(self):
        return str(self.data)

    def __mul__(self, other):
        op = Mul()
        result = op.forward(self, other)

        if self.requires_grad:
            result.grad_fn = op

        return result

    def backward(self):
        self.grad_fn.backward(1)


    def shape(self):
        return self.data.shape

class Mul:
    ctx = None

    def forward(self, tensor, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(tensor.data * other.data, requires_grad= tensor.requires_grad or other.requires_grad)
        result.is_leaf = False

        self.ctx = [tensor, other]

        return result

    def backward(self, grad_output):
        tensor = self.ctx[0]
        other = self.ctx[1]

        grad_tensor = other.data * grad_output
        grad_other = tensor.data * grad_output

        if tensor.requires_grad:
            if tensor.grad_fn is None:
                tensor.grad += grad_tensor
            else:
                tensor.grad_fn.backward(grad_tensor)

        if other.requires_grad:
            if other.grad_fn is None:
                other.grad += grad_other
            else:
                other.grad_fn.backward(grad_other)


def main():
    x = Tensor([2, 2], requires_grad=True)
    y = Tensor([1, 3])
    z = x * y
    print(z.grad_fn)
    z.backward()
    print(x.grad)


if __name__ == "__main__":
    main()