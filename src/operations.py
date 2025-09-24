import numpy as np
from src.base import Function


class Function(Function):
    """Base class for all operations.
    """
    ctx = None

    def _update_grad(self, tensor, grad):
        """Updates the grad of a tensor if necessary

        Checks if the tensor is a leaf:
        - If it is, the grad needs to be accumulated in the node
        - If not,the grad needs to propagated to the node's father

        Args:
            tensor (Tensor): The input tensor to update the grad for.
            grad (np.ndarray): The gradient to accumulate.
        """

        if tensor.requires_grad:
            if tensor.is_leaf:
                tensor.grad = np.add(tensor.grad, grad) if tensor.grad is not None else grad
            else:
                tensor.grad_fn.backward(grad)


    def _result_tensor(self, value, requires_grad):
        """Creates a Tensor with the values provided
        This function is called in the forward method of each operation, 
        so the resulting tensor is not a leaf

        Args:
            value (np.ndarray): The data for the resulting tensor.
            requires_grad (bool): Whether the resulting tensor requires gradients.

        Returns:
            Tensor: The resulting tensor.
        """

        from src.tensor import Tensor

        result = Tensor(value, requires_grad=requires_grad)
        result.is_leaf = False

        return result


class Mul(Function):
    def forward(self, tensor, other):
        """Multiplies two tensors element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
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
        """Adds two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(tensor.data + other.data, tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (x + a) d/dx = 1
        (x + a) d/da = 1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, grad_output)

class Sub(Function):
    def forward(self, tensor, other):
        """"Subtracts two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
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
        """Raises a tensor to the power of index element-wise.

        Returns:
            Tensor: The resulting tensor.
        """

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

        with np.errstate(divide='ignore', invalid='ignore'): # Avoid numpy warnings with extreme values
            tensor_grad = index.data * (np.pow(tensor.data, (index.data - 1))) * grad_output
            index_grad = result * np.log(tensor.data) * grad_output

        self._update_grad(tensor, tensor_grad)
        self._update_grad(index, index_grad)

class Dot(Function):
    def forward(self, tensor, other):
        """Performs a dot product between two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
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
        """Sums all elements in a tensor.

        Returns:
            Tensor: The resulting tensor.
        """

        self.ctx = tensor

        return self._result_tensor(tensor.data.sum(), tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (sum(x)) d/dx = np.ones(x.shape)
        """
        tensor = self.ctx

        self._update_grad(tensor, np.ones(tensor.shape()) * grad_output)

class Transpose(Function):
    def forward(self, tensor):
        """Transposes a tensor.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = tensor

        return self._result_tensor(tensor.data.T, tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (x.T) d/dx = I
        """
        tensor = self.ctx

        self._update_grad(tensor, grad_output.T)

class Maximum(Function):
    def forward(self, tensor, other):
        """Returns the element-wise maximum of two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(np.maximum(tensor.data, other.data), tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads
        (max(x, a)) d/dx = 1 if x > a else 0
        (max(x, a)) d/da = 1 if a > x else 0
        """

        tensor, other = self.ctx

        tensor_grad = tensor.data > other.data
        other_grad = other.data > tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Minimum(Function):
    def forward(self, tensor, other):
        """Returns the element-wise minimum of two tensors.
        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(np.minimum(tensor.data, other.data), tensor.requires_grad or other.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        (min(x, a)) d/dx = 1 if x < a else 0
        (min(x, a)) d/da = 1 if a < x else
        """
        tensor, other = self.ctx

        tensor_grad = tensor.data < other.data
        other_grad = other.data < tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Div(Function):
    def forward(self, tensor, other):
        """Divides two tensors element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
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

class SigmoidOp(Function):
    def forward(self, tensor):
        """Applies the sigmoid activation function element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
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
        """Applies the softmax activation function to a 1D tensor.

        Returns:
            Tensor: The resulting tensor.
        """ 
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