import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx


def broadcast_spec(a: NDArray, b: NDArray) -> NDArray:
    return a + b

@kernel
def broadcast_test1(a: NDArray, b: NDArray) -> NDArray:
    out = np.zeros((a.shape[0], b.shape[1]))
    for i, j in np.ndindex(out.shape):
        out[i, j] = a[i, 0] + b[0, j]

    return out

@kernel
def broadcast_test2(a: NDArray, b: NDArray, out: NDArray)->None:
    i = threadIdx.x
    j = threadIdx.y
    if i >= a.shape[0] or j >= b.shape[1]:
        return

    out[i, j] = a[i, 0] + b[0, j]


if __name__ == "__main__":
    SIZE = 2
    a = np.arange(SIZE).reshape(SIZE, 1)
    b = np.arange(SIZE).reshape(1, SIZE)

    out2 = np.zeros((SIZE, SIZE))
    broadcast_test2[(1, 1), (3, 3)](a, b, out2)

    out1 = broadcast_test1[(1, 1), (3, 3)](a, b)

    out_spec = broadcast_spec(a, b)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)