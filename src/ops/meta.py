from src.ops import cpu

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

    if A.shape() != B.shape():
        pass #TODO

    out = A.empty_like()

    if A.device == "cpu":
        return cpu.add_cpu(A.data, B.data, out)
    elif A.device == "cuda":
        pass
        # kernels.add_kernel(a, b, out)
    else:
        raise NotImplementedError
