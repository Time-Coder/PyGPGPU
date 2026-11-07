"""Bug: a body literal is truncated to float32 under float=np.float64."""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel(float=np.float64)
def double_body_literal(out: np.ndarray):
    """A 17-digit literal is a python double; the device should read it as one."""
    for i in range(out.shape[0]):
        value = 0.1234567890123456789
        out[i] = value
