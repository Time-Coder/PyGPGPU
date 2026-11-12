"""Bug: min/max refuse more than two arguments."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def min_with_three_arguments(x: np.float32, out: np.ndarray):
    """`min(a, b, c)` -- python's builtin takes any arity, left to right."""
    for i in range(out.shape[0]):
        out[i] = min(x, 0.7, 0.9) + max(x, 0.2, 0.1)
