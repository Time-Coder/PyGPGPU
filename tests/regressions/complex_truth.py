"""Bug: `bool(z)` on a complex value fails to compile."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def complex_truthiness(out: np.ndarray):
    """`if z:` on a complex value is `z != 0` in python (either part non-zero)."""
    for i in range(out.shape[0]):
        z0 = complex(0.0, 0.0)
        z1 = complex(1.0, 0.0)
        out[i, 0] = 1.0 if z0 else 0.0
        out[i, 1] = 1.0 if z1 else 0.0
