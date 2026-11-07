"""Bug: a device helper cannot return two scalars."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


def _helper_returns_two(a: np.float64, b: np.float64):
    """A device helper answering two scalars, the way a python function would."""
    return np.float64(a + b), np.float64(a - b)


@kernel
def unpacks_two_scalars(params: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        first, second = _helper_returns_two(params[0], params[1])
        out[i, 0] = first
        out[i, 1] = second
