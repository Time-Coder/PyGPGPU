import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx, blockIdx, blockDim, zeros_per_block, barrier


def map_spec(a: NDArray) -> NDArray:
    return a + 10

@kernel
def shared_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    shared = zeros_per_block((4,), np.float32)
    for i in range(out.shape[0]):
        local_i = threadIdx.x
        shared[blockIdx][local_i] = a[i]
        barrier()
        out[i] = shared[blockIdx][local_i] + 10

    return out

@kernel
def shared_test2(a: NDArray, out: NDArray) -> None:
    shared = zeros_per_block((4,), np.float32)
    i = blockIdx.x * blockDim.x + threadIdx.x
    local_i = threadIdx.x

    if i >= a.shape[0]:
        return

    shared[blockIdx][local_i] = a[i]
    barrier()
    
    out[i] = shared[blockIdx][local_i] + 10


if __name__ == "__main__":
    SIZE = 4
    a = np.arange(SIZE)

    out_spec = map_spec(a)

    out1 = shared_test1[(2, 1), (SIZE, 1)](a)

    out2 = np.zeros_like(a)
    shared_test2[(2, 1), (SIZE, 1)](a, out2)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)