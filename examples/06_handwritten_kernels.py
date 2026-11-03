"""06_handwritten_kernels.py -- import a handwritten .cl file like a Python module.

`kernel_import` is the import hook for hand-written kernels. One `.cl` file with no
siblings is a module with exactly one backend, so it is the single-backend case of
`15_kernel_import.py`: the kernel is used like a `@kernel` function, `["opencl"]` names the
backend it has, and a bracket naming any other backend raises KeyError -- there is no
implementation there.

`examples/vecadd.cl` also doubles as the source `07_low_level_oop.py` compiles through the
low-level `Context.compile` API, which needs no import hook at all.

Run:
    python examples/06_handwritten_kernels.py
"""
import os
import sys

import numpy as np

from pygpgpu import kernel_import, list_devices

# Make the import find vecadd.cl in this example's directory
_ex_dir = os.path.dirname(os.path.abspath(__file__))
if _ex_dir not in sys.path:
    sys.path.insert(0, _ex_dir)

kernel_import.install()
import vecadd                       # the module named by vecadd.cl


def main():
    # 1) Plain array vector add + scalar offset
    n = 1_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    out = np.empty_like(a)
    # Work size is inferred automatically from the read-back argument's shape.
    # Every kernel also takes the real element count so it can clamp: the launch
    # size may be rounded up to a multiple of the local size.
    vecadd.add_offset(a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    # Manually specified (blocks per grid, threads per block); the launch must cover the data.
    out.fill(0.0)
    vecadd.add_offset[(3907, 256)](a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset (auto + manual work size) OK")

    # 2) The one backend this module has, named explicitly
    out.fill(0.0)
    vecadd.add_offset["opencl"](a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset on opencl OK")

    # 3) A backend the file does not implement raises KeyError at the bracket
    assert vecadd.add_offset.backends == ["opencl"]
    assert vecadd.add_offset.sources()["opencl"].endswith("vecadd.cl")
    for selector in ("cuda", "hip"):
        try:
            vecadd.add_offset[selector](a, b, out, np.float32(1.0), np.int32(n))
        except KeyError as error:
            print(f"add_offset on {selector} raised KeyError as expected: {error}")

    # 4) struct as a parameter and as an array
    Pt = vecadd.Pt                       # a ctypes Structure
    pts = np.array([Pt(x=3.0, y=1.0), Pt(x=0.5, y=0.5)], dtype=Pt)  # the numpy.array patch supports nested Structure
    res = np.zeros(2, dtype=np.int32)
    center = Pt(x=0.5, y=0.5)
    vecadd.in_circle(pts, res, center, np.float32(1.0), np.int32(2))
    assert res.tolist() == [0, 1], res
    print("in_circle (struct argument) OK:", res.tolist())

    print("devices on this machine:", list_devices())


if __name__ == "__main__":
    main()
