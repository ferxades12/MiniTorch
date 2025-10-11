"""CPU implementations of tensor operations.
"""

import numpy as np

def add_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.add(A, B, out=out)

def mul_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.multiply(A, B, out=out)

def sub_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.subtract(A, B, out=out)

def pow_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.pow(A, B, out=out)

def dot_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.dot(A, B, out=out)

def div_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.divide(A, B, out=out)

def sum_cpu(A: np.ndarray, axis, out: np.ndarray) -> np.ndarray:
    return np.sum(A, axis=axis, out=out)

def abs_cpu(A: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.absolute(A, out=out)

def transpose_cpu(A: np.ndarray) -> np.ndarray:
    return np.transpose(A)

def maximum_cpu(A: np.ndarray, B:np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.maximum(A, B, out=out)

def minimum_cpu(A: np.ndarray, B:np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.minimum(A, B, out=out)

def log_cpu(A: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.log(A, out=out)



