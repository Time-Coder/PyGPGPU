import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx, blockIdx, blockDim, zeros_per_block, barrier

TPB = 8

def pool_spec(a: NDArray):
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()

    return out

@kernel
def pool_test1(a: NDArray) -> NDArray:
    out = np.zeros_like(a)
    shared = zeros_per_block((TPB,), np.float32)
    for i in range(out.shape[0]):
        local_i = threadIdx.x
        shared[blockIdx][local_i] = a[i]
        barrier()
        sum_value: float = 0.0
        for j in range(3):
            local_sub_i = local_i - j
            if local_sub_i >= 0:
                sum_value += shared[blockIdx][local_sub_i]
        out[i] = sum_value

    return out

@kernel
def pool_test2(a: NDArray, out: NDArray) -> None:
    shared = zeros_per_block((TPB,), np.float32)
    i = blockIdx.x * blockDim.x + threadIdx.x
    local_i = threadIdx.x

    if i >= a.shape[0]:
        return

    shared[blockIdx][local_i] = a[i]
    barrier()
    
    sum_value: float = 0.0
    for j in range(3):
        local_sub_i = local_i - j
        if local_sub_i >= 0:
            sum_value += shared[blockIdx][local_sub_i]
    out[i] = sum_value


if __name__ == "__main__":
    SIZE = 8
    a = np.arange(SIZE)

    out_spec = pool_spec(a)

    out1 = pool_test1[(1, 1), (TPB, 1)](a)

    out2 = np.zeros_like(a)
    pool_test2[(1, 1), (TPB, 1)](a, out2)

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)