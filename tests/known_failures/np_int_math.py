"""Bug: np.* on integer literals is computed in float32 (numpy answers float64)."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def numpy_math_on_ints(out: np.ndarray):
    """`np.sqrt(2)` is a numpy float64 in python; the device emits sqrtf((float)2)."""
    for i in range(out.shape[0]):
        out[i, 0] = np.sqrt(2)
        out[i, 1] = np.log(10)
        out[i, 2] = np.exp(1)
