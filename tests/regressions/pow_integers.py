"""Bug: `pow(2, 10)` (python builtin, int arguments) fails to compile on CUDA and
OpenCL. HIP's clang happens to resolve the same call, so the bug is backend-dependent."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def builtin_pow_integers(out: np.ndarray):
    """`pow(2, 10)` is 1024; the device emits `pow((int)2, (int)10)`, which NVRTC and
    OpenCL's clang call ambiguous -- HIP's `<cmath>` resolves it, so the identical source
    compiles there and refuses elsewhere. python's builtin answers an int (exact); the
    device spelling should cast to float the way the `**` operator already does."""
    for i in range(out.shape[0]):
        out[i] = pow(2, 10)
