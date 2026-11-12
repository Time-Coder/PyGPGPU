"""Bug: a scalar dtype cast inside the magic loop is refused as a creation."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def scalar_cast_inside_the_loop(cre: np.float64, out: np.ndarray):
    """`two = np.float64(2.0)` inside the loop body -- a cast, not an array creation."""
    for i in range(out.shape[0]):
        two = np.float64(2.0)
        z = cre
        for _ in range(3):
            z = z * z
        out[i] = z * two
