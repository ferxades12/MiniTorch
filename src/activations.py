from src.operations import Function
import src as M

def ReLU(tensor):
    return M.maximum(tensor, 0)