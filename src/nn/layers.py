from src.nn.module import Module
import numpy as np
from src.tensor import Tensor, stack
from src.ops import DropoutOp
from src.nn.activations import Tanh, ReLU

class Linear(Module):
    def __init__(self, in_features:int, out_features:int, bias:bool=True):
        """_summary_

        Args:
            in_features (int): Number of input features (weights)
            out_features (int): Number of output features (neurons)
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__()
        k = np.sqrt(1 / in_features)
        self.weight:Tensor = _initialize_parameter(k, (in_features, out_features))

        self.params.append(self.weight)

        if bias:
            self.bias:Tensor = _initialize_parameter(k, out_features)
            self.params.append(self.bias)
        else:
            self.bias = None
       
    def forward(self, x:Tensor) -> Tensor:
        """Applies a linear transformation to the incoming data: y = xW + b
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features)

        Raises:
            ValueError: If the input tensor's last dimension does not match in_features.
        """

        if x.shape()[-1] != self.weight.shape()[0]:
            raise ValueError("Variable dimensions are not correct")
        
        x = x if isinstance(x, Tensor) else Tensor(x)
        bias = self.bias if self.bias is not None else 0

        return x.dot(self.weight) + bias
    
class Dropout(Module):
    def __init__(self, p:float = 0.5):
        """Dropout regularization layer.

        Args:
            p (float, optional): Probability of dropping out a neuron. Defaults to 0.5
        """
        super().__init__()
        self.p = p
    
    def forward(self, x:Tensor) -> Tensor:
        """Applies dropout to the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with dropout applied.
        """
        return x._apply_unary_op(DropoutOp, self.p, self.training)


class Sequential(Module):
    def __init__(self, *args : Module):
        super().__init__()
        

        for module in args:
            if not isinstance(module, Module):
                raise ValueError("All arguments must be instances of Module")

            self.submodules.append(module)
       
    def forward(self, x:Tensor) -> Tensor:
        for module in self.submodules:
            x = module(x)
        
        return x
    
    def add_module(self, module:Module) -> None:
        if not isinstance(module, Module):
            raise ValueError("Argument must be an instance of Module")
        
        self.submodules.append(module)

class RNN(Module):
    """A single-layer RNN where:
    output_size = hidden_size
    """
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh", dropout=0):
        """A single-layer RNN 

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            nonlinearity (str, optional): Activation function to use. Defaults to "tanh".

        Raises:
            ValueError: If nonlinearity is not "tanh" or "relu".
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if dropout == 0:
            self.dropout_layer = None
        else:
            self.dropout_layer = Dropout(dropout)

        if nonlinearity == "tanh":
            self.func = Tanh()
        elif nonlinearity == "relu":
            self.func = ReLU()
        else:
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        
        k = np.sqrt(1 / hidden_size)
        self.Wx = _initialize_parameter(k, (input_size, hidden_size))
        self.Wh = _initialize_parameter(k, (hidden_size, hidden_size))
        self.Wy = _initialize_parameter(k, (hidden_size, hidden_size))
        self.bh = _initialize_parameter(k, hidden_size)
        self.by = _initialize_parameter(k, hidden_size)

        self.params.extend([self.Wx, self.Wh, self.Wy, self.bh, self.by])

    def forward(self, x:Tensor):
        """Forward pass for the simple RNN.

        For each timestep, computes the hidden state and output using the following equations:

        h_(t) = func(x_(t) @ Wx + h_(t-1) @ Wh + bh) : (hidden_size, batch_size)

        x_t : (batch_size, input_size)
        Wx : (input_size, hidden_size)

        h_t : (batch_size, hidden_size)
        Wh : (hidden_size, hidden_size)

        bh : (hidden_size,) 

        ------------------------------------------------------------------------
        
        y_t = func(Wy @ ht + by) : (hidden_size, batch_size)

        h_t : (batch_size, hidden_size)
        Wy : (hidden_size, hidden_size)
        by : (hidden_size,)

        ------------------------------------------------------------------------

        For every layer, processes the entire sequence and passes the output to the next layer.
        The output grows in depth, but the sequence length and batch size remain constant.


        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
                Where:
                - batch_size: number of sequences in the batch
                - seq_len: number of timesteps (length of the sequence)
                - input_size: Number of features at each timestep

        Returns:
            List[Tensor]: List of output tensors for each timestep, each of shape (batch_size, hidden_size).
            Tensor: Final hidden state tensor of shape (batch_size, hidden_size)
        """

        for layer_idx in range(self.num_layers):

            batch_size, seq_len, input_size = x.shape()

            h = [None for _ in range(seq_len + 1)]
            h[0] = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=True)

            for t in range(seq_len):
                x_t = Tensor(x[:, t, :], requires_grad=True)

                h[t+1] = self.func(x_t.dot(self.Wx) + h[t].dot(self.Wh) + self.bh)

                #y_t = self.func(h[t+1].dot(self.Wy) + self.by)
            
            x = stack(h[1:])

            if self.dropout_layer != None and layer_idx != self.num_layers - 1:
                x = self.dropout_layer(x)

        return x, h[-1]





def _initialize_parameter(k, size):
    return Tensor(np.random.uniform(-1 * k, k, size), requires_grad=True)