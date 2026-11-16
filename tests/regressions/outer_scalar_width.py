"""Bug: an outer-scope integer bypasses the width check a kernel *argument* gets.

A python `int` has no width of its own, so it travels as the width the kernel chose for
`int` (32 bits by default). A value outside that width is refused with a `ValueError` naming
the numpy spelling that would carry it -- but only for a *parameter*. A module-level or
closure constant is inlined into the generated C source by the translator and never passes
through `_check_integer_fits`, so it wraps silently: the device reads `-2147483648` where
python wrote `2147483648`.

`WIDE_CONSTANT` is the value; `uses_wide_constant` inlines it, while `uses_wide_argument`
receives the same number as a parameter (the control that already refuses).
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel

# 2**31: one past the largest value an `int` (32 bits) can hold.
WIDE_CONSTANT: int = 2 ** 31


@kernel
def uses_wide_constant(out: np.ndarray):
    """`out[i] = WIDE_CONSTANT` -- inlined, so the width check never ran."""
    for i in range(out.shape[0]):
        out[i] = WIDE_CONSTANT


@kernel
def uses_wide_argument(out: np.ndarray, value: int):
    """The same number as a parameter, which the launch already refuses."""
    for i in range(out.shape[0]):
        out[i] = value
