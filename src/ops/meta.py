from src.ops import cpu
import numpy as np

""" Dispatch table for operations """
DISPATCH_TABLE = {
    "add":{
        "cpu" : cpu.add_cpu,
        "gpu" : None
    },
    "mul":{
        "cpu" : cpu.mul_cpu,
        "gpu" : None
    },
    "sub":{
        "cpu" : cpu.sub_cpu,
        "gpu" : None
    },
    "pow":{
        "cpu" : cpu.pow_cpu,
        "gpu" : None
    },
    "div":{
        "cpu" : cpu.div_cpu,
        "gpu" : None
    },
    "sum":{
        "cpu" : cpu.sum_cpu,
        "gpu" : None
    },
    "abs":{
        "cpu" : cpu.abs_cpu,
        "gpu" : None
    },
    "transpose":{
        "cpu" : cpu.transpose_cpu,
        "gpu" : None
    },
    "maximum":{
        "cpu" : cpu.maximum_cpu,
        "gpu" : None
    },
    "minimum":{
        "cpu" : cpu.minimum_cpu,
        "gpu" : None
    },
    "log":{
        "cpu" : cpu.log_cpu,
        "gpu" : None
    },
}



def add(A, B) -> np.ndarray:
    """Adds two tensors element-wise, supporting broadcasting.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise sum of A and B.
    """
    return _dispatch("add", *_prepare_bitwise_op(A, B))


def mul(A, B) -> np.ndarray:
    """Multiplies two tensors element-wise, supporting broadcasting.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise product of A and B.
    """
    return _dispatch("mul", *_prepare_bitwise_op(A, B))

def sub(A, B) -> np.ndarray:
    """Subtracts two tensors element-wise, supporting broadcasting.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise difference of A and B.
    """
    return _dispatch("sub", *_prepare_bitwise_op(A, B))

def pow(A, B) -> np.ndarray:
    """Raises each element of A to the power of the corresponding element in B.

    Args:
        A (Tensor): The base tensor.
        B (Tensor): The exponent tensor.
    Returns:
        (np.ndarray): The element-wise power of A raised to B.
    """
    return _dispatch("pow", *_prepare_bitwise_op(A, B))

def dot(A, B) -> np.ndarray:
    """Performs a dot product between two 2D tensors

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The result of the dot product.
    """
    a = A.shape()
    b = B.shape()

    assert a[1] == b[0]

    shape = a[0], b[1]
    out = np.empty(shape)

    return _dispatch("dot", A.device, A, B, out)

def maximum(A, B) -> np.ndarray:
    """Computes the element-wise maximum of two tensors.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise maximum of A and B.
    """
    return _dispatch("maximum", A.device, *_prepare_bitwise_op(A,B))

def minimum(A, B) -> np.ndarray:
    """Computes the element-wise minimum of two tensors.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise minimum of A and B.
    """
    return _dispatch("minimum", A.device, *_prepare_bitwise_op(A,B))


def div(A, B) -> np.ndarray:
    """Divides two tensors element-wise, supporting broadcasting.

    Args:
        A (Tensor): The first input tensor.
        B (Tensor): The second input tensor.
    Returns:
        (np.ndarray): The element-wise quotient of A and B.
    """
    return _dispatch("div", A.device, *_prepare_bitwise_op(A, B))


def sum(A, axis=None) -> np.ndarray:
    """Sums the elements of a tensor along specified axes.

    Args:
        A (Tensor): The input tensor.
        axis (Union[int, list[int], None]): The axis or axes along which to sum.
    Returns:
        (np.ndarray): The sum of the elements along the specified axes.
    """
    if axis is None:
        out = np.empty(())
    elif isinstance(axis, list):
        shape = A.shape()
        
        for ax in axis:
            shape[ax] = 1
        
        out = np.empty(shape)
    else:
        shape = A.shape()
        shape[axis] = 1
        out = np.empty(shape)
    
    return _dispatch("sum", A.device, A, axis, out)   

def abs(A) -> np.ndarray:
    """Computes the element-wise absolute value of a tensor.

    Args:
        A (Tensor): The input tensor.
    Returns:
        (np.ndarray): The element-wise absolute value of A.
    """
    return _dispatch("abs", A.device, A, np.empty_like(A))


def transpose(A) -> np.ndarray:
    """Transposes a 2-D tensor by reversing the order of its dimensions.

    Args:
        A (Tensor): The input tensor.
    Returns:
        (np.ndarray): The transposed array.
    """
    return _dispatch("transpose", A.device, A, (A.shape[1], A.shape[0]))

def log(A) -> np.ndarray:
    """Computes the element-wise natural logarithm of a tensor.
    Args:
        A (Tensor): The input tensor.
    Returns:
        (np.ndarray): The element-wise natural logarithm of A.
    """
    return _dispatch("log", A.device, A, np.empty_like(A))

    
def _prepare_bitwise_op(A, B) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checks and prepares two tensors for a bitwise operation.

    Args:
        A (np.ndarray): The first input array.
        B (np.ndarray): The second input array.

    Returns:
        (str, np.ndarray, np.ndarray, np.ndarray): The device and the prepared input arrays and the output array.
    """

    assert A.device == B.device

    if A.shape != B.shape:
        A_data, B_data = broadcast(A.data, B.data)
    else:
        A_data, B_data = A.data, B.data

    out = np.empty_like(A_data)

    return A.device, A_data, B_data, out



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

    if device not in DISPATCH_TABLE[op]:
        raise NotImplementedError(f"Operation {op} no supported on device {device}")

    return DISPATCH_TABLE[op][device](*args)


def broadcast(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]: #TODO mover a utils
    """Broadcasts two arrays to a common shape.
    Args:
        A (np.ndarray): First array.
        B (np.ndarray): Second array.
    Returns:
        (np.ndarray, np.ndarray): The broadcasted arrays.
    """
    shape = np.broadcast_shapes(A.shape, B.shape)

    return np.broadcast_to(A, shape), np.broadcast_to(B, shape)
