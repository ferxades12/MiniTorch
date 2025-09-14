import numpy as np

class Tensor:
    grad = None
    grad_fn = None
    is_leaf = False

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad

    def __str__(self):
        return str(self.data)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data)
        return result

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result =  other.data * self.data
        return result

    def shape(self):
        return self.data.shape

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data @ other.data


def main():
    x = Tensor([2, 2])
    y = Tensor(([2], [2]))
    print(x.dot(y))


if __name__ == "__main__":
    main()