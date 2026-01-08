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

def sigmoid_cpu(A: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.divide(1.0, (1.0 + np.exp(-A)), out=out)

def softmax_cpu(A: np.ndarray, axis: int, out: np.ndarray) -> np.ndarray:
    np.subtract(A, np.max(A, axis=axis, keepdims=True), out=out) # Normalize to avoid overflow
    np.exp(out, out=out)
    np.divide(out, np.sum(out, axis=axis, keepdims=True), out=out)
    
    return out

def cross_entropy_one_hot_cpu(softmax: np.ndarray, target: np.ndarray) -> float:
    # target * log(softmax), luego sum por clase, luego mean por batch
    return -np.mean(np.sum(target * np.log(softmax), axis=1))


def cross_entropy_indices_cpu(softmax: np.ndarray, target: np.ndarray) -> float:
    # np.arange(len(target.data)), target.data.astype(int)] = [i, target[i]] with i = 1,...,len(target.data)
    result = softmax[np.arange(len(target)), target.astype(int)]
    return -np.mean(np.log(result))
    