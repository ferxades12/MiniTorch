import numpy as np


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
    def _result_tensor(self, value, requires_grad):
        """
        Creates a Tensor with the values provided
        """

        from src.tensor import Tensor

        result = Tensor(value, requires_grad=requires_grad)
        result.is_leaf = False

        return result


class Mul(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(tensor.data * other.data, tensor.requires_grad or other.requires_grad)

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
        self.ctx = (tensor, other)

        return self._result_tensor(tensor.data + other.data, tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (x + a) d/dx = 1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, grad_output)

class Sub(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(tensor.data - other.data, tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (x - a) d/dx = -1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, -1 * grad_output)

class Pow(Function):
    def forward(self, tensor, index):
        result = self._result_tensor(np.pow(tensor.data, index.data), tensor.requires_grad or index.requires_grad)

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

class Dot(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(np.dot(tensor.data, other.data), tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (A * X) d/dx =
        (X * A) d/dx =
        """

        tensor, other = self.ctx

        tensor_grad = np.dot(grad_output, other.data.T)
        other_grad = np.dot(tensor.data.T, grad_output)

        self._update_grad(tensor, tensor_grad)
        self._update_grad(other, other_grad)

class Sum(Function):
    def forward(self, tensor):
        self.ctx = tensor

        return self._result_tensor(tensor.data.sum(), tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads
        """
        tensor = self.ctx

        self._update_grad(tensor, np.ones(tensor.shape()) * grad_output)

class Traspose(Function):
    def forward(self, tensor):
        self.ctx = tensor

        return self._result_tensor(tensor.data.T, tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads
        """
        tensor = self.ctx

        self._update_grad(tensor, grad_output.T)

class Maximum(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(np.maximum(tensor.data, other.data), tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        tensor, other = self.ctx

        tensor_grad = tensor.data > other.data
        other_grad = other.data > tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Minimum(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(np.minimum(tensor.data, other.data), tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        tensor, other = self.ctx

        tensor_grad = tensor.data < other.data
        other_grad = other.data < tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Div(Function):
    def forward(self, tensor, other):
        self.ctx = (tensor, other)

        return self._result_tensor(tensor.data / other.data, tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        x / a d/dx = 1/a
        a / x d/dx = -a/x^2
        """

        tensor, other = self.ctx

        tensor_grad = (1 / other.data) * grad_output
        other_grad = (-tensor.data / (other.data**2)) * grad_output

        self._update_grad(tensor, tensor_grad)
        self._update_grad(other, other_grad)

class Sigmoid(Function):
    def forward(self, tensor):
        sigmoid = 1 / (1 + np.exp(-1 * tensor.data))
        self.ctx = (tensor, sigmoid)

        return self._result_tensor(sigmoid, tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        y = sigmoid(x)
        dy/dx = y * (1 - y)
        """

        tensor, sigmoid = self.ctx

        tensor_grad = sigmoid * (1 - sigmoid) * grad_output

        self._update_grad(tensor, tensor_grad)

class Softmax1D(Function):
    def forward(self, tensor):  
        if tensor.numdims() != 1:
            raise ValueError("Softmax1D only accepts 1D tensors")

        # Normalization needs to be done to avoid overflow
        t_stable = tensor.data - np.max(tensor.data)

        exps = np.exp(t_stable)
        softmax = exps / np.sum(exps)

        self.ctx = (tensor, softmax)

        return self._result_tensor(softmax, tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        y = softmax(x)
        dy/dx = diag(y) - y y^T
        """

        tensor, softmax = self.ctx

        s = softmax.reshape(-1, 1) # To ensure the dims

        tensor_grad = np.diagflat(s) - np.dot(s, s.T)

        self._update_grad(tensor, np.dot(tensor_grad, grad_output))