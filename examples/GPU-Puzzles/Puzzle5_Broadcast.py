import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel


def broadcast_spec(a: NDArray, b: NDArray) -> NDArray:
    return a + b

@kernel
def broadcast_test(a: NDArray, b: NDArray) -> NDArray:
    out = np.zeros((a.shape[0], b.shape[1]))
    for i, j in np.ndindex(out.shape):
        out[i, j] = a[i, 0] + b[0, j]

    return out


SIZE = 2
a = np.arange(SIZE).reshape(SIZE, 1)
b = np.arange(SIZE).reshape(1, SIZE)

assert np.all(broadcast_test(a, b) == broadcast_spec(a, b))