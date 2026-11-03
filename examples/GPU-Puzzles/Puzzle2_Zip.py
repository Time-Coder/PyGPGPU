import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx


def zip_spec(a: NDArray, b: NDArray) -> NDArray:
    return a + b


@kernel
def zip_test1(a: NDArray, b: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i in np.ndindex(out.shape[0]):
        out[i] = a[i] + b[i]

    return out


@kernel
def zip_test2(a: NDArray, b: NDArray, out: NDArray) -> None:
    i = threadIdx.x
    out[i] = a[i] + b[i]


if __name__ == "__main__":
    SIZE = 4
    a = np.arange(SIZE)
    b = np.arange(SIZE)

    out_spec = zip_spec(a, b)
    out1 = zip_test1[(1, 1), (SIZE, 1)](a, b)

    out2 = np.zeros_like(a)
    zip_test2[(1, 1), (SIZE, 1)](a, b, out2)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)