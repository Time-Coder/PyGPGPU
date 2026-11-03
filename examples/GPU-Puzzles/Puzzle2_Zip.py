import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel


def zip_spec(a: NDArray, b: NDArray) -> NDArray:
    return a + b

@kernel
def zip_test(a: NDArray, b: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i in np.ndindex(out.shape[0]):
        out[i] = a[i] + b[i]

    return out


SIZE = 4
a = np.arange(SIZE)
b = np.arange(SIZE)
assert np.all(zip_test(a, b) == zip_spec(a, b))