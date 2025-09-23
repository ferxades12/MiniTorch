import numpy as np
import src as M

def ReLU(tensor):
    return M.maximum(tensor, 0)

def sigmoid(tensor):
    return M.sigmoid(tensor)

def tanh(tensor):
    return 2 * sigmoid(2 * tensor) - 1

def softmax(tensor):
    return M.softmax(tensor)