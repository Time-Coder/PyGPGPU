import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel


def map_2D_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_2D_test(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i, j in np.ndindex(out.shape):
        out[i, j] = a[i, j] + 10

    return out


SIZE = 2
out = np.zeros((SIZE, SIZE))
a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))

assert np.all(map_2D_test(a) == map_2D_spec(a))