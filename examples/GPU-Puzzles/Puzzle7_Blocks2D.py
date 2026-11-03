import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx, blockIdx, blockDim


def map_block2D_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_block2D_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i, j in np.ndindex(a.shape):
        out[i, j] = a[i, j] + 10

    return out

@kernel
def map_block2D_test2(a: NDArray, out: NDArray)->None:
    i = blockIdx.x * blockDim.x + threadIdx.x
    j = blockIdx.y * blockDim.y + threadIdx.y
    if i >= a.shape[0] or j >= a.shape[1]:
        return

    out[i, j] = a[i, j] + 10


if __name__ == "__main__":
    SIZE = 5
    out2 = np.zeros((SIZE, SIZE))
    a = np.ones((SIZE, SIZE))

    out1 = map_block2D_test1[(2, 2), (3, 3)](a)
    map_block2D_test2[(2, 2), (3, 3)](a, out2)

    out_spec = map_block2D_spec(a)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)