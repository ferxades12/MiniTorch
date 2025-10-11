from src.ops import cpu
import numpy as np

"""
a + b
a.__add__(b)
Add()(a, b)
ops.add(a,b) # Meta. Validacion de shapes y tipos, broadcasting, dispatching


.backward()
dividir si es costosa


"""
def add(A, B):
    assert A.device == B.device
    assert hasattr(A, "grad") and hasattr(B, "grad")

    if A.shape != B.shape:
        A_data, B_data = broadcast(A.data, B.data)
    else:
        A_data, B_data = A.data, B.data

    out = np.empty_like(A_data)

    if A.device == "cpu":
        return cpu.add_cpu(A_data, B_data, out)
    elif A.device == "cuda":
        pass
        # kernels.add_kernel(a, b, out)
    else:
        raise NotImplementedError


def broadcast(A, B):
    shape = np.broadcast_shapes(A.shape, B.shape)

    return np.broadcast_to(A, shape), np.broadcast_to(B, shape)