import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx


def map_guard_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_guard_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i in range(out.shape[0]):
        out[i] = a[i] + 10

    return out

@kernel
def map_guard_test2(a: NDArray, out: NDArray) -> None:
    i = threadIdx.x
    if i >= a.shape[0]:
        return

    out[i] = a[i] + 10


if __name__ == "__main__":
    SIZE = 4
    a = np.arange(SIZE)

    out_spec = map_guard_spec(a)
    out1 = map_guard_test1[(1, 1), (8, 1)](a)

    out2 = np.zeros_like(a)
    map_guard_test2[(1, 1), (8, 1)](a, out2)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)