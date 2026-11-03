import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx, blockIdx, blockDim


def map_block_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_block_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = a[i] + 10

    return out

@kernel
def map_block_test2(a: NDArray, out: NDArray)->None:
    i = blockIdx.x * blockDim.x + threadIdx.x
    if i >= a.shape[0]:
        return

    out[i] = a[i] + 10


if __name__ == "__main__":
    SIZE = 9
    out2 = np.zeros((SIZE,))
    a = np.arange(SIZE)

    out1 = map_block_test1[(3, 1), (4, 1)](a)
    map_block_test2[(3, 1), (4, 1)](a, out2)

    out_spec = map_block_spec(a)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)