from src.ops import cpu, cuda
import numpy as np
import cupy as cp

""" Dispatch table for operations """
DISPATCH_TABLE = {
    "add":{
        "cpu" : cpu.add_cpu,
        "cuda" : cuda.add_cuda
    },
    "mul":{
        "cpu" : cpu.mul_cpu,
        "cuda" : cuda.mul_cuda
    },
    "sub":{
        "cpu" : cpu.sub_cpu,
        "cuda" : cuda.sub_cuda
    },
    "pow":{
        "cpu" : cpu.pow_cpu,
        "cuda" : cuda.pow_cuda
    },
    "div":{
        "cpu" : cpu.div_cpu,
        "cuda" : cuda.div_cuda
    },
    "sum":{
        "cpu" : cpu.sum_cpu,
        "cuda" : cuda.sum_cuda
    },
    "abs":{
        "cpu" : cpu.abs_cpu,
        "cuda" : cuda.abs_cuda
    },
    "transpose":{
        "cpu" : cpu.transpose_cpu,
        "cuda" : cuda.transpose_cuda
    },
    "maximum":{
        "cpu" : cpu.maximum_cpu,
        "cuda" : cuda.maximum_cuda
    },
    "minimum":{
        "cpu" : cpu.minimum_cpu,
        "cuda" : cuda.minimum_cuda
    },
    "log":{
        "cpu" : cpu.log_cpu,
        "cuda" : cuda.log_cuda
    },
    "dot":{
        "cpu" : cpu.dot_cpu,
        "cuda" : cuda.dot_cuda
    },
    "sigmoid":{
        "cpu" : cpu.sigmoid_cpu,
        "cuda" : cuda.sigmoid_cuda
    },
    "softmax":{
        "cpu" : cpu.softmax_cpu,
        "cuda" : cuda.softmax_cuda
    },
    "cross_entropy_one_hot":{
        "cpu" : cpu.cross_entropy_one_hot_cpu,
        "cuda" : cuda.cross_entropy_one_hot_cuda
    },
    "cross_entropy_indices":{
        "cpu" : cpu.cross_entropy_indices_cpu,
        "cuda" : cuda.cross_entropy_indices_cuda
    },
}


def _apply_bitwise_op(op:str, A, B) -> np.ndarray:
    """Checks and prepares two tensors for a bitwise operation, then applies the operation.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
        op (str): The operation to perform.

    Returns:
        np.ndarray: The result of the operation.
    """
    assert A.device == B.device
    xp = cp if A.device == "cuda" else np

    if A.shape != B.shape:
        A_data, B_data = _broadcast(A.data, B.data, xp)
    else:
        A_data, B_data = A.data, B.data

    out = xp.empty_like(A_data)

    return _dispatch(op, A.device, A_data, B_data, out)


def _apply_unary_op(op:str, A, out=None) -> np.ndarray:
    """Applies a unary operation to a tensor.

    Args:
        op (str): The operation to perform.
        A (Tensor): The input tensor.
        out (np.ndarray, optional): Pre-allocated output array.

    Returns:
        np.ndarray: The result of the operation.
    """
    xp = cp if A.device == "cuda" else np
    if out is None:
        out = xp.empty_like(A.data)
    return _dispatch(op, A.device, A.data, out)


def _apply_dot(A, B) -> np.ndarray:
    """Performs a dot product between two tensors.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.

    Returns:
        np.ndarray: The result of the dot product.
    """
    assert A.device == B.device
    a = A.shape()
    b = B.shape()

    if A.numdims() == 1 and B.numdims() == 1:
        assert a[0] == b[0]
        shape = ()
    elif A.numdims() == 2 and B.numdims() == 1:
        assert a[1] == b[0]
        shape = (a[0],)
    elif A.numdims() == 1 and B.numdims() == 2:
        assert a[0] == b[0]
        shape = (b[1],)
    elif A.numdims() == 2 and B.numdims() == 2:
        assert a[1] == b[0]
        shape = (a[0], b[1])
    else:
        raise ValueError("Dot product only supports 1D or 2D tensors.")

    out = np.empty(shape, dtype=np.result_type(A.data, B.data)) # np.dot is picky with dtypes

    return _dispatch("dot", A.device, A.data, B.data, out)


def _apply_sum(A, axis=None) -> np.ndarray:
    """Sums the elements of a tensor along specified axes.

    Args:
        A (Tensor): The input tensor.
        axis (Union[int, list[int], None]): The axis or axes along which to sum.

    Returns:
        np.ndarray: The sum of the elements.
    """
    xp = cp if A.device == "cuda" else np
    if axis is None:
        out = xp.empty(())
    elif isinstance(axis, list):
        shape = list(A.shape())
        for ax in axis:
            shape.pop(ax)
        out = xp.empty(shape)
    else:
        shape = list(A.shape())
        shape.pop(axis)
        out = xp.empty(shape)
    
    return _dispatch("sum", A.device, A.data, axis, out)


def _apply_transpose(A) -> np.ndarray:
    """Transposes a 2-D tensor.

    Args:
        A (Tensor): The input tensor.

    Returns:
        np.ndarray: The transposed array.
    """
    return _dispatch("transpose", A.device, A.data)


def _apply_softmax(A) -> np.ndarray:
    """Applies softmax activation to a tensor.

    Args:
        A (Tensor): The input tensor.

    Returns:
        np.ndarray: The softmax result.
    """
    xp = cp if A.device == "cuda" else np
    # Meta: determinar axis según dimensiones
    axis = None if A.numdims() == 1 else 1
    
    # Allocate output
    out = xp.empty_like(A.data)
    
    # Dispatch al kernel con axis
    return _dispatch("softmax", A.device, A.data, axis, out)

def _apply_cross_entropy_one_hot(softmax, target) -> float:
    """Computes the cross-entropy loss with one-hot encoded targets.

    Args:
        softmax (Tensor): The softmax probabilities (already computed).
        target (Tensor): The true target values (one-hot encoded).

    Returns:
        float: The computed cross-entropy loss (scalar).
    """
    assert softmax.device == target.device
    return _dispatch("cross_entropy_one_hot", softmax.device, softmax.data, target.data)


def _apply_cross_entropy_indices(softmax, target) -> float:
    """Computes the cross-entropy loss with class index targets.

    Args:
        softmax (Tensor): The softmax probabilities (already computed).
        target (Tensor): The true target values (class indices).

    Returns:
        float: The computed cross-entropy loss (scalar).
    """
    assert softmax.device == target.device
    return _dispatch("cross_entropy_indices", softmax.device, softmax.data, target.data)


def _dispatch(op:str, device:str, *args) -> np.ndarray:
    """Dispatches the operation to the appropriate device function.

    Args:
        op (str): The operation to perform.
        device (str): The device to use ('cpu' or 'cuda').

    Raises:
        NotImplementedError: If the operation is not supported on the given device.

    Returns:
        (np.ndarray): The result of the dispatched operation.
    """

    if device not in DISPATCH_TABLE[op] or DISPATCH_TABLE[op][device] is None:
        raise NotImplementedError(f"Operation '{op}' not supported on device '{device}'")

    return DISPATCH_TABLE[op][device](*args)


def _broadcast(A: np.ndarray, B: np.ndarray, xp) -> tuple[np.ndarray, np.ndarray]: #TODO mover a utils
    """Broadcasts two arrays to a common shape.
    Args:
        A (np.ndarray): First array.
        B (np.ndarray): Second array.
    Returns:
        (np.ndarray, np.ndarray): The broadcasted arrays.
    """
    shape = xp.broadcast_shapes(A.shape, B.shape)

    return xp.broadcast_to(A, shape), xp.broadcast_to(B, shape)