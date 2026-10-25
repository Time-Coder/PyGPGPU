"""06_handwritten_kernels.py -- import handwritten .cl / .cu / .hip files like
Python modules.

Once the matching import hook (climport / cuimport / hipimport) is installed,
`import vecadd` treats `vecadd.cl` as a module: every kernel and struct type
in it becomes a module attribute, and a `.pyi` stub is generated alongside for
IDE completion. The hooks for the three backends work identically (this
example uses OpenCL, available on essentially every platform; switching to
`.cu/.hip` requires the CUDA/HIP runtime on the machine).

The hooks load lazily: the corresponding runtime is initialized only when
OpenCL is actually used.

Run:
    python examples/06_handwritten_kernels.py
"""
import sys
import os
import numpy as np
from pygpgpu import climport  # cuimport / hipimport work the same way

# Make the import find vecadd.cl in this example's directory
_ex_dir = os.path.dirname(os.path.abspath(__file__))
if _ex_dir not in sys.path:
    sys.path.insert(0, _ex_dir)

climport.install()
climport.config()          # accepts BuildOptions (e.g. fp32 precision options)
import vecadd              # just like importing a Python module


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
    # Manually specified (total work items, per group); global must cover the data.
    out.fill(0.0)
    vecadd.add_offset[(1_000_192, 256)](a, b, out, np.float32(1.0), np.int32(n))
    assert np.allclose(out, a + b + 1.0)
    print("add_offset (auto + manual work size) OK")

    # 2) struct as a parameter and as an array
    Pt = vecadd.Pt                       # a ctypes Structure
    pts = np.array([Pt(x=3.0, y=1.0), Pt(x=0.5, y=0.5)], dtype=Pt)  # the numpy.array patch supports nested Structure
    res = np.zeros(2, dtype=np.int32)
    center = Pt(x=0.5, y=0.5)
    vecadd.in_circle(pts, res, center, np.float32(1.0), np.int32(2))
    assert res.tolist() == [0, 1], res
    print("in_circle (struct argument) OK:", res.tolist())


if __name__ == "__main__":
    main()