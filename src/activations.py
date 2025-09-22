import numpy as np
import src as M

def ReLU(tensor):
    return M.maximum(tensor, 0)

def sigmoid(tensor):
    return M.sigmoid(tensor)