from __future__ import annotations

import numpy as np
from pygpgpu import (kernel)



@kernel
def equality(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == b[i] else 0.0


@kernel
def equality_with_real(a: np.ndarray, value: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] == value else 0.0


@kernel
def less_than(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] < b[i] else 0.0
