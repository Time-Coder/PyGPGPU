from __future__ import annotations

import numpy as np
from pygf import conjugate
from pygpgpu import (kernel)



@kernel
def read_members(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = a[i].real
        out[i, 1] = a[i].imag


@kernel
def conjugate_method(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i].conjugate()


@kernel
def conjugate_call(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = conjugate(a[i])


@kernel
def magnitude(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = abs(a[i])


@kernel
def phase(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.angle(a[i])


@kernel
def complex_sqrt(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sqrt(a[i])


@kernel
def complex_exp(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp(a[i])


@kernel
def complex_log(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log(a[i])


@kernel
def complex_log2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log2(a[i])


@kernel
def complex_log10(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.log10(a[i])


@kernel
def complex_sin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sin(a[i])


@kernel
def complex_cos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cos(a[i])


@kernel
def complex_tan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tan(a[i])


@kernel
def complex_asin(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arcsin(a[i])


@kernel
def complex_acos(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arccos(a[i])


@kernel
def complex_atan(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.arctan(a[i])


@kernel
def complex_sinh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.sinh(a[i])


@kernel
def complex_cosh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.cosh(a[i])


@kernel
def complex_tanh(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.tanh(a[i])


@kernel
def complex_square(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.square(a[i])


@kernel
def complex_reciprocal(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.reciprocal(a[i])


@kernel
def complex_exp2(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = np.exp2(a[i])


COMPLEX_FUNCTIONS = {
    "sqrt": complex_sqrt, "exp": complex_exp, "log": complex_log,
    "log2": complex_log2, "log10": complex_log10,
    "sin": complex_sin, "cos": complex_cos, "tan": complex_tan,
    "asin": complex_asin, "acos": complex_acos, "atan": complex_atan,
    "sinh": complex_sinh, "cosh": complex_cosh, "tanh": complex_tanh,
    "square": complex_square, "reciprocal": complex_reciprocal,
    "exp2": complex_exp2,
}
