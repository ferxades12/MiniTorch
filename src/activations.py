import numpy as np
import src as M

def ReLU(tensor):
    return M.maximum(tensor, 0)

def sigmoid(tensor):
    return M.sigmoid(tensor)

def tanh(tensor):
    return 2 * sigmoid(2 * tensor) - 1

#def softmax(tensor):
    """
    Applies the softmax function to the input tensor along the last dimension.

    The softmax function is defined as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    """