import numpy as np
from src.base import Function, Array
from src.ops import dispatch

try:
    import cupy as cp
except ImportError:
    cp = None


class OpFunction(Function):
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
            grad (Array): The gradient to accumulate.
        """

        if tensor.requires_grad:
            if tensor.shape() != grad.shape:
                grad = self._unbroadcast(grad, tensor.shape())

            if tensor.is_leaf:
                tensor.grad = tensor.grad + grad if tensor.grad is not None else grad
            else:
                tensor.grad_fn.backward(grad)

    def _result_tensor(self, value : Array, requires_grad: bool, device: str = "cpu"):
        """Creates a Tensor with the values provided
        This function is called in the forward method of each operation, 
        so the resulting tensor is not a leaf

        Args:
            value (Array): The data for the resulting tensor.
            requires_grad (bool): Whether the resulting tensor requires gradients.
            device (str): Device to create the tensor on ('cpu' or 'cuda').

        Returns:
            Tensor: The resulting tensor.
        """

        from src.tensor import Tensor
    
        result = Tensor(value, requires_grad=requires_grad, device=device)
        result.is_leaf = False

        return result

    def _unbroadcast(self, grad, target_shape):
        """Unbroadcasts the gradient to match the target shape.

            If the grad has more dimensions than the target shape, it means
            grad was broascasted, so we sum over the extra dimensions.

            If a dimension in the target has size 1, that dimension was broadcasted, 
            so we sum over it in the gradient to match the target shape.
        Args:
            grad (Array): The gradient to unbroadcast.
            target_shape (tuple): The target shape to match.

        Returns:
            Array: The unbroadcasted gradient.
        """
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)


        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(target_shape)

class Mul(OpFunction):
    def forward(self, tensor, other):
        """Multiplies two tensors element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("mul", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the multiplication operation.

        ax d/dx = a
        """

        tensor, other = self.ctx

        self._update_grad(tensor,  other.data * grad_output)
        self._update_grad(other, tensor.data *grad_output)

class Add(OpFunction):
    def forward(self, tensor, other):
        """Adds two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("add", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the addition operation.

        (x + a) d/dx = 1
        (x + a) d/da = 1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, grad_output)

class Sub(OpFunction):
    def forward(self, tensor, other):
        """"Subtracts two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("sub", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the subtraction operation.

        (x - a) d/dx = -1
        """

        tensor, other = self.ctx

        self._update_grad(tensor, grad_output)
        self._update_grad(other, -1 * grad_output)

class Pow(OpFunction):
    def forward(self, tensor, index):
        """Raises a tensor to the power of index element-wise.

        Returns:
            Tensor: The resulting tensor.
        """

        result = self._result_tensor(dispatch._apply_bitwise_op("pow", tensor, index), tensor.requires_grad or index.requires_grad, tensor.device)

        self.ctx = (tensor, index, result.data)

        return result

    def backward(self, grad_output):
        """
        Computes the gradient of the power operation.

        x^a d/dx = a * x^(a-1)
        a^x d/dx = a^x * ln(a)
        """
        tensor, index, result = self.ctx
        xp = tensor.xp

        tensor_grad = index.data * (xp.power(tensor.data, (index.data - 1))) * grad_output

        safe_data = xp.where(tensor.data <= 0, 1e-10, tensor.data)
        index_grad = result * xp.log(safe_data) * grad_output

        self._update_grad(tensor, tensor_grad)
        self._update_grad(index, index_grad)

class Dot(OpFunction):
    def forward(self, tensor, other):
        """Performs a dot product between two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_dot(tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the dot product operation.

        (A * X) d/dx = A.T
        (X * A) d/dx = X.T
        """

        tensor, other = self.ctx
        xp = tensor.xp

        tensor_grad = xp.dot(grad_output, other.data.T)
        other_grad = xp.dot(tensor.data.T, grad_output)

        self._update_grad(tensor, tensor_grad)
        self._update_grad(other, other_grad)

class Sum(OpFunction):
    def forward(self, tensor, axis=None):
        """Sums all elements in a tensor.

        Returns:
            Tensor: The resulting tensor.
        """

        self.ctx = tensor, axis

        return self._result_tensor(dispatch._apply_sum(tensor, axis=axis), tensor.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the sum operation.

        (sum(x)) d/dx = np.ones(x.shape)
        """
        tensor, ax = self.ctx
        xp = tensor.xp
        # TODO refactorizar
        if ax is None:
            grad = xp.ones(tensor.shape()) * grad_output
        else:
            # Expand dims to match the original tensor shape
            grad = xp.expand_dims(grad_output, axis=ax)
            grad = xp.broadcast_to(grad, tensor.shape())
        self._update_grad(tensor, grad)

class Abs(OpFunction):
    def forward(self, tensor):
        self.ctx = tensor

        return self._result_tensor(dispatch._apply_unary_op("abs", tensor), tensor.requires_grad, tensor.device)
    
    def backward(self, grad_output):
        tensor = self.ctx
        xp = tensor.xp

        tensor_grad = xp.where(tensor.data > 0, 1, 0)  + xp.where(tensor.data < 0, -1, 0)
        self._update_grad(tensor, tensor_grad * grad_output)

        
class Transpose(OpFunction):
    def forward(self, tensor):
        """Transposes a tensor.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = tensor

        return self._result_tensor(dispatch._apply_transpose(tensor), tensor.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the transpose operation.

        (x.T) d/dx = I
        """
        tensor = self.ctx

        self._update_grad(tensor, grad_output.T)

class Maximum(OpFunction):
    def forward(self, tensor, other):
        """Returns the element-wise maximum of two tensors.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("maximum", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the maximum operation.

        (max(x, a)) d/dx = 1 if x > a else 0
        (max(x, a)) d/da = 1 if a > x else 0
        """

        tensor, other = self.ctx

        tensor_grad = tensor.data > other.data
        other_grad = other.data > tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Minimum(OpFunction):
    def forward(self, tensor, other):
        """Returns the element-wise minimum of two tensors.
        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("minimum", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the minimum operation.

        (min(x, a)) d/dx = 1 if x < a else 0
        (min(x, a)) d/da = 1 if a < x else
        """
        tensor, other = self.ctx

        tensor_grad = tensor.data < other.data
        other_grad = other.data < tensor.data

        self._update_grad(tensor, tensor_grad * grad_output)
        self._update_grad(other, other_grad * grad_output)

class Div(OpFunction):
    def forward(self, tensor, other):
        """Divides two tensors element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
        self.ctx = (tensor, other)

        return self._result_tensor(dispatch._apply_bitwise_op("div", tensor, other), tensor.requires_grad or other.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the division operation.

        x / a d/dx = 1/a
        a / x d/dx = -a/x^2
        """

        tensor, other = self.ctx

        tensor_grad = (1 / other.data) * grad_output
        other_grad = (-tensor.data / (other.data**2)) * grad_output

        self._update_grad(tensor, tensor_grad)
        self._update_grad(other, other_grad)

class Log(OpFunction):
    def forward(self, tensor):

        self.ctx = tensor

        return self._result_tensor(dispatch._apply_unary_op("log", tensor), tensor.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the logarithm function.

        log x d/dx = 1/x
        """

        tensor = self.ctx

        tensor_grad = (1 / tensor.data) * grad_output

        self._update_grad(tensor, tensor_grad)

class SigmoidOp(OpFunction):
    def forward(self, tensor):
        """Applies the sigmoid activation function element-wise.

        Returns:
            Tensor: The resulting tensor.
        """
        sigmoid = dispatch._apply_unary_op("sigmoid", tensor)
        self.ctx = (tensor, sigmoid)

        return self._result_tensor(sigmoid, tensor.requires_grad, tensor.device)

    def backward(self, grad_output):
        """
        Computes the gradient of the sigmoid function.

        y = sigmoid(x)
        dy/dx = y * (1 - y)
        """

        tensor, sigmoid = self.ctx

        tensor_grad = sigmoid * (1 - sigmoid) * grad_output

        self._update_grad(tensor, tensor_grad)

class SoftmaxOp(OpFunction):
    def forward(self, tensor): 
        """Applies the softmax activation function to a tensor.

        Returns:
            Tensor: The resulting tensor.
        """ 
        softmax = dispatch._apply_softmax(tensor)
        
        self.ctx = (tensor, softmax)

        return self._result_tensor(softmax, tensor.requires_grad, tensor.device)


    def backward(self, grad_output):
        """
        Computes the gradient of the softmax function.

        s = softmax(x)
        dL/dx = (diag(s) - s @ s.T) * dL/ds
        """
        tensor, s = self.ctx
        xp = tensor.xp

        if tensor.numdims() == 1:
            # Case 1D: Calculates the gradient directly
            s = s.reshape(-1, 1)  
            tensor_grad = xp.dot(xp.diagflat(s) - xp.dot(s, s.T), grad_output)
        else:
            # Otherwise: Calculates the gradient row by row
            tensor_grad = xp.zeros_like(tensor.data)
            for i in range(len(tensor.data)):
                si = s[i].reshape(-1, 1)  
                tensor_grad[i] = xp.dot(xp.diagflat(si) - xp.dot(si, si.T), grad_output[i])

        self._update_grad(tensor, tensor_grad)

class CrossEntropyOp(OpFunction):
    def forward(self, prediction, target):
        """
        Calculates the Cross-Entropy loss between prediction and target tensors.

        If target is a vector Ex. [2, 1, 0]
            CE = - Σ log(Softmax(prediction)[i, target[i]])  .mean()
            This selects the correct class based on the index in target for each sample
        
        If target is a one-hot tensor Ex. [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            CE = - Σ (target * log(Softmax(prediction)))   .mean()
            The bit-wise multiplication implicitly selects the correct class for each sample
        
        Args:
            prediction (Tensor): The predicted logits tensor.
            target (Tensor): The target tensor (one-hot or class indices).

        Returns:
            Tensor: The resulting scalar tensor after calculating the Cross-Entropy loss.
        """
        
        s = SoftmaxOp().forward(prediction.copy())

        if target.shape() == prediction.shape():
            # Target is one-hot
            self.ctx = (prediction, s, target, True)

            result = dispatch._apply_cross_entropy_one_hot(s, target)
        else:
            # Target is a vector
            self.ctx = (prediction, s, target, False)

            result = dispatch._apply_cross_entropy_indices(s, target)
        
        return self._result_tensor(result, prediction.requires_grad, prediction.device)
    
    def backward(self, grad_output):
        """Computes the gradient of the Cross-Entropy loss.

        - Σ (target * log(Softmax(x)))   d/dx               = (Softmax(x) - target) / batchsize
        - Σ log(Softmax(prediction)[i, target[i]])   d/dx   = (Softmax(x) - 1 if i = target[i] else 0) / batchsize
        """
        prediction, s, target, is_one_hot = self.ctx

        prediction_grad = (s - (target if is_one_hot else target.one_hot(s.shape()[1]))) / prediction.shape()[0]
        
        self._update_grad(prediction, prediction_grad.data * grad_output)

class DropoutOp(OpFunction):
    def forward(self, x, p: float, training:bool = True):
        """Calculates

        Args:
            x (Tensor): Tensor to apply dropout to
            p (float): probability of a neuron to be shut down
            training (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        
        if training:
            mask = x.xp.random.binomial(1, 1 - p, x.shape())
            result = x.data * mask / (1 - p)
        else:
            mask = None
            result = x.data

        self.ctx = (x, mask, p)
        return self._result_tensor(result, x.requires_grad, x.device)
    
    def backward(self, grad_output):
        """ Computes the gradient of the Dropout operation.

        (Dropout(x, p)) d/dx = mask / (1-p)
        """
        x, mask, p = self.ctx

        if mask is None:
            grad = grad_output
        else:
            grad = mask / (1-p) * grad_output
        self._update_grad(x, grad)
