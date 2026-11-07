"""Bug: a float carried scalar is accumulated through an int atomic."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def float_carried_sum(x: np.ndarray, out: np.ndarray):
    """`total = 0.0` above the loop, `total += x[i]` inside it.

    The carried-scalar machinery declares the slot `int*` and emits `atomicAdd(total,
    x[i])` -- an int atomic handed a double value -- so the contributions are truncated
    to int and the sum loses every fractional part. The interpreter answers python's own
    sum, so the two backends disagree on the same source.
    """
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i]
    out[0] = total
