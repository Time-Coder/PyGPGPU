"""Bug: float-spelled transcendentals under float=np.float64."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel(float=np.float64)
def double_math_functions(x: float, out: np.ndarray):
    """sin/cos/sqrt/exp at the mapped width -- a python float argument travels as the
    mapping's double, so every one of these must compute in double."""
    for i in range(out.shape[0]):
        out[i, 0] = np.sin(x)
        out[i, 1] = np.cos(x)
        out[i, 2] = np.sqrt(x)
        out[i, 3] = np.exp(x)
