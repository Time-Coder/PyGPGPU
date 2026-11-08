from __future__ import annotations

import numpy as np
from pygpgpu import (kernel)

from .constants import (
    SCOPE_128,
    SCOPE_64,
    SCOPE_PY,
)


@kernel
def passed_scalar(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value


@kernel
def returned_element(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i]


@kernel
def returned_computed(a: np.ndarray):
    for i in range(a.shape[0]):
        return a[i] ** 2


@kernel
def returned_constant(out: np.ndarray):
    return 3 + 4j


@kernel
def read_scope_python(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_PY


@kernel
def read_scope_complex64(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_64


@kernel
def read_scope_complex128(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = SCOPE_128


@kernel
def return_scope(out: np.ndarray):
    return SCOPE_128


@kernel
def return_scope_complex64(out: np.ndarray):
    return SCOPE_64
