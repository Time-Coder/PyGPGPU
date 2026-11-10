import numpy as np
from numpy.typing import NDArray
from pygpgpu import kernel, threadIdx, blockIdx, blockDim, shared_array_per_block, barrier

TPB = 8

def dot_spec(a: NDArray, b: NDArray)->NDArray:
    return a @ b

@kernel
def dot_test1(a: NDArray, b: NDArray) -> NDArray:
    out = np.zeros(1)
    shared = shared_array_per_block((TPB,), np.float32)
    for i in range(a.shape[0]):
        local_i = threadIdx.x
        shared[blockIdx][local_i] = a[i] * b[i]
        barrier()

        if local_i == 0:
            sum_value: float = 0.0
            for local_sub_i in range(TPB):
                sum_value += shared[blockIdx][local_sub_i]

            out[0] = sum_value

    return out

@kernel
def dot_test2(a: NDArray, b: NDArray, out: NDArray) -> None:
    shared = shared_array_per_block((TPB,), np.float32)
    i = blockIdx.x * blockDim.x + threadIdx.x
    local_i = threadIdx.x

    if i >= a.shape[0]:
        return

    shared[blockIdx][local_i] = a[i] * b[i]
    barrier()

    if local_i == 0:
        sum_value: float = 0.0
        for local_sub_i in range(TPB):
            sum_value += shared[blockIdx][local_sub_i]

        out[0] = sum_value


if __name__ == "__main__":
    SIZE = 8
    a = np.arange(SIZE)
    b = np.arange(SIZE)

    out_spec = dot_spec(a, b)

    out1 = dot_test1[(1, 1), (SIZE, 1)](a, b)

    out2 = np.zeros(1)
    dot_test2[(1, 1), (SIZE, 1)](a, b, out2)

    print(f"spec {out_spec}  magic-loop {out1[0]}  gid {out2[0]}")

    assert np.all(out1 == out_spec)
    assert np.all(out2 == out_spec)
