"""CUDA implementations of tensor operations using CuPy.
"""

import cupy as cp

def add_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.add(A, B, out=out)

def mul_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.multiply(A, B, out=out)

def sub_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.subtract(A, B, out=out)

def pow_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.power(A, B, out=out)

def dot_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.dot(A, B, out=out)

def div_cuda(A: cp.ndarray, B: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.divide(A, B, out=out)

def sum_cuda(A: cp.ndarray, axis, out: cp.ndarray) -> cp.ndarray:
    return cp.sum(A, axis=axis, out=out)

def abs_cuda(A: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.absolute(A, out=out)

def transpose_cuda(A: cp.ndarray) -> cp.ndarray:
    return cp.transpose(A)

def maximum_cuda(A: cp.ndarray, B:cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.maximum(A, B, out=out)

def minimum_cuda(A: cp.ndarray, B:cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.minimum(A, B, out=out)

def log_cuda(A: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.log(A, out=out)

def sigmoid_cuda(A: cp.ndarray, out: cp.ndarray) -> cp.ndarray:
    return cp.divide(1.0, (1.0 + cp.exp(-A)), out=out)

def softmax_cuda(A: cp.ndarray, axis: int, out: cp.ndarray) -> cp.ndarray:
    cp.subtract(A, cp.max(A, axis=axis, keepdims=True), out=out) # Normalize to avoid overflow
    cp.exp(out, out=out)
    cp.divide(out, cp.sum(out, axis=axis, keepdims=True), out=out)
    
    return out

def cross_entropy_one_hot_cuda(softmax: cp.ndarray, target: cp.ndarray) -> float:
    # target * log(softmax), luego sum por clase, luego mean por batch
    return -cp.mean(cp.sum(target * cp.log(softmax), axis=1))


def cross_entropy_indices_cuda(softmax: cp.ndarray, target: cp.ndarray) -> float:
    # cp.arange(len(target.data)), target.data.astype(int)] = [i, target[i]] with i = 1,...,len(target.data)
    result = softmax[cp.arange(len(target)), target.astype(int)]
    return -cp.mean(cp.log(result))
    