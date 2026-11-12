"""Bug: int arithmetic inside the body wraps silently (python does not).

Two shapes, two mechanisms:

* a product of two *literals* (`5000000 * 1000000`) folds at translation time, so the
  exact value can be range-checked like a scalar argument;
* a product with a *parameter* (`step * 1000000`) is a launch-time value, and the
  existing escape hatch is the `int=np.int64` mapping -- the fix pins that it answers
  python's number instead of wrapping.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def literal_multiply_overflows(out: np.ndarray):
    """`5000000 * 1000000` -- both literals, so the exact value is checkable."""
    for i in range(out.shape[0]):
        big = 5000000 * 1000000
        out[i] = big


@kernel(int=np.int64)
def parameter_multiply_wide(step: int, out: np.ndarray):
    """`step * 1_000_000` under `int=np.int64` -- the documented 64-bit spelling."""
    for i in range(out.shape[0]):
        big = step * 1000000
        out[i] = big
