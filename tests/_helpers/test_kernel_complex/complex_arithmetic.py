from __future__ import annotations

import numpy as np
from pygpgpu import (kernel)

from .constants import (
    SCOPE_128,
)


@kernel
def add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def subtract(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - b[i]


@kernel
def multiply(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * b[i]


@kernel
def divide(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / b[i]


@kernel
def scale(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * factor


@kernel
def scale_div(a: np.ndarray, factor: np.float32, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / factor


@kernel
def scale_by_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


@kernel
def add_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + 2.0


@kernel
def sub_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] - 2.0


@kernel
def div_by_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] / 2.0


@kernel
def pow_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2.0


@kernel
def rsub_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 - a[i]


@kernel
def rdiv_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 / a[i]


@kernel
def rpow_float(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = 2.0 ** a[i]


@kernel
def scalar_scale(value, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = value * 3.0


@kernel
def negate(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = -a[i]


@kernel
def identity(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = +a[i]


@kernel
def square_by_operator(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 2


@kernel
def cube(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] ** 3


@kernel
def mixed_add(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def multiply_by_scope(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * SCOPE_128
