import numpy as np

def add_cpu(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.add(A, B, out=out)
