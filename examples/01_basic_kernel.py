"""01_basic_kernel.py -- the most basic @kernel usage: SAXPY.

One Python source drives all three backends (OpenCL / CUDA / HIP):
- `@kernel` decorates an ordinary python function;
- the outermost `for i in range(<array>.shape[0])` loop becomes the work-item
  loop, and the work size (total thread count) defaults to that array's first
  dimension;
- with no backend specified, the first available one is probed in the order
  `cuda -> hip -> opencl`.

Run:
    python examples/01_basic_kernel.py
"""
import numpy as np
from pygpgpu import kernel


@kernel
def saxpy(x, y, out, alpha):
    """SAXPY: out[i] = x[i] * alpha + y[i]."""
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


def main():
    n = 1_000_000
    # Use float32; @kernel does not support fp16 on CUDA/HIP
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    out = np.zeros_like(y)

    # 1) No backend specified: auto-probe (cuda -> hip -> opencl)
    saxpy(x, y, out, 2.0)
    assert np.allclose(out, x * 2.0 + y)
    print("Default backend:", saxpy.backend, "- OK")

    # 2) Explicit backend via brackets (opencl / cuda / hip; aliases ocl/cu/rocm/amd)
    out.fill(0.0)
    saxpy["cuda"](x, y, out, 2.0)
    assert np.allclose(out, x * 2.0 + y)
    print("cuda        - OK")

    # 3) backend:device form in brackets (device = index or name substring)
    out.fill(0.0)
    saxpy["opencl:0"](x, y, out, 2.0)
    assert np.allclose(out, x * 2.0 + y)
    print("opencl:0    - OK")

    # 4) Manual work size (total, per group); global is the total work-item
    #    count; CUDA/HIP grid block count = global / local, converted automatically
    out.fill(0.0)
    saxpy[(128 * 4096, 256)](x, y, out, 2.0)
    assert np.allclose(out, x * 2.0 + y)
    print("Manual work size - OK")


if __name__ == "__main__":
    main()