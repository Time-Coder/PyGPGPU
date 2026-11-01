import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel


def map_guard_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def map_guard_test(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    for i in range(out.shape[0]):
        out[i] = a[i] + 10

    return out


SIZE = 4
a = np.arange(SIZE)
assert np.all(map_guard_test(a) == map_guard_spec(a))