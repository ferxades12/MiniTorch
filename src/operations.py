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
    def forward(self, tensor, axis=None):
        """Sums all elements in a tensor.

        Returns:
            Tensor: The resulting tensor.
        """

        self.ctx = tensor

        return self._result_tensor(np.sum(tensor.data, axis=axis), tensor.requires_grad)

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

class Log(Function):
    def forward(self, tensor):

        self.ctx = tensor

        return self._result_tensor(np.log(tensor.data), tensor.requires_grad)

    def backward(self, grad_output):
        """
        Retrieves the data in ctx and updates grads

        log x d/dx = 1/x
        """

        tensor = self.ctx

        tensor_grad = (1 / tensor.data) * grad_output

        self._update_grad(tensor, tensor_grad)

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

class SoftmaxOp(Function):
    def forward(self, tensor): 
        """Applies the softmax activation function to a tensor.

        Returns:
            Tensor: The resulting tensor.
        """ 
        axis = None if tensor.numdims() == 1 else 1

        t_stable = tensor.data - np.max(tensor.data, axis=axis, keepdims=True) # Normalize to avoid overflow

        exps = np.exp(t_stable)
        softmax = exps / np.sum(exps, axis=axis, keepdims=True)
        
        self.ctx = (tensor, softmax)

        return self._result_tensor(softmax, tensor.requires_grad)


    def backward(self, grad_output):
        """
        Computes the gradient of the softmax function.

        s = softmax(x)
        dL/dx = (diag(s) - s @ s.T) * dL/ds
        """
        tensor, s = self.ctx

        if tensor.numdims() == 1:
            # Case 1D: Calculates the gradient directly
            s = s.reshape(-1, 1)  
            tensor_grad = np.dot(np.diagflat(s) - np.dot(s, s.T), grad_output)
        else:
            # Otherwise: Calculates the gradient row by row
            tensor_grad = np.zeros_like(tensor.data)
            for i in range(len(tensor.data)):
                si = s[i].reshape(-1, 1)  # Convierte cada fila en columna
                tensor_grad[i] = np.dot(np.diagflat(si) - np.dot(si, si.T), grad_output[i])

        self._update_grad(tensor, tensor_grad)


class CrossEntropyOp(Function):
    def forward(self, prediction, target):
        """
        Calculates the Cross-Entropy loss between prediction and target tensors.

        If target is a vector Ex. [2, 1, 0]
            CE = - Σ log(Softmax(prediction)[i, target[i]])
            This selects the correct class based on the index in target for each sample
        
        If target is a one-hot tensor Ex. [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            CE = - Σ (target * log(Softmax(prediction)))
            The bit-wise multiplication implicitly selects the correct class for each sample
        
        Args:
            prediction (Tensor): The predicted logits tensor.
            target (Tensor or list): The ground truth target tensor (one-hot or class indices).

        Returns:
            Tensor: The resulting scalar tensor after calculating the Cross-Entropy loss.
        """
        
        s = SoftmaxOp()(prediction)

        if target.shape() == s.shape():
            # Target is one-hot
            self.ctx = (prediction, s, target, True)

            return -1 * (target * s.log()).sum(axis=1)
            

        else:
            # Target is a vector
            self.ctx = (prediction, s, target, False)

            # np.arange(len(target.data)), target.data.astype(int)] = [i, target[i]] with i = 1,...,len(target.data)
            return -1 * s[np.arange(len(target.data)), target.data.astype(int)]
    
    def backward(self, grad_output):
        """_summary_

        - Σ (target * log(Softmax(x)))   d/dx               = Softmax(x) - target
        - Σ log(Softmax(prediction)[i, target[i]])   d/dx   = Softmax(x) - 1 if i = target[i] else 0
        """
        prediction, s, target, is_one_hot = self.ctx

        prediction_grad = s - (target if is_one_hot else target.one_hot(s.shape()[0])) # No se si es 0 o 1
        
        self._update_grad(prediction, prediction_grad * grad_output)
