import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx


def map_2D_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_2D_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i, j in np.ndindex(out.shape):
        out[i, j] = a[i, j] + 10

    return out

@kernel
def map_2D_test2(a: NDArray, out: NDArray) -> NDArray:
    i = threadIdx.x
    j = threadIdx.y
    if i >= a.shape[0] or j >= a.shape[1]:
        return

    out[i, j] = a[i, j] + 10


if __name__ == "__main__":
    SIZE = 2
    out2 = np.zeros((SIZE, SIZE))
    a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))

    out_spec = map_2D_spec(a)
    out1 = map_2D_test1[(1, 1), (3, 3)](a)
    map_2D_test2[(1, 1), (3, 3)](a, out2)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)