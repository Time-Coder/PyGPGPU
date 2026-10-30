"""The scalar math intrinsics, and the fact that pygf's spelling works in a kernel.

`ray.py` writes `sqrt(delta)` after importing `sqrt` from pygf, and that used to be translated
as an ordinary *device helper*: the translator then compiled pygf's own body
(`_single_op(x, math.sqrt, "sqrt")`) and failed on `math.sqrt is not supported`. The name was
simply missing from the builtin set -- pygf publishes about ninety functions and only a
fraction were exposed, so the same would have happened for `sin`, `pow` and the rest.

A builtin name that pygf also publishes *is* pygf's object (see the note in AGENTS.md), so the
host value and the device intrinsic agree by construction. This file pins the values on all
four backends, which is also what proves each `device_builtins` row compiles where it is used.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import (
    acos, asin, atan, atan2, cbrt, ceil, cos, cosh, exp, fabs, floor, fmod, hypot,
    kernel, log, log10, log2, pow, sin, sinh, sqrt, tan, tanh, trunc,
)

# One input value per function, and what it must produce. A kernel body is real source, so the
# calls are written out: `inspect.getsource` has to be able to read the function.
UNARY = [
    ("sqrt", 4.0, 2.0),
    ("sin", 0.0, 0.0),
    ("cos", 0.0, 1.0),
    ("tan", 0.0, 0.0),
    ("asin", 0.5, 0.5235988),
    ("acos", 0.5, 1.0471976),
    ("atan", 1.0, 0.7853982),
    ("sinh", 0.0, 0.0),
    ("cosh", 0.0, 1.0),
    ("tanh", 0.0, 0.0),
    ("exp", 0.0, 1.0),
    ("log", 1.0, 0.0),
    ("log2", 8.0, 3.0),
    ("log10", 100.0, 2.0),
    ("floor", 2.7, 2.0),
    ("ceil", 2.1, 3.0),
    ("trunc", -2.7, -2.0),
    ("fabs", -3.5, 3.5),
    ("cbrt", 8.0, 2.0),
]

BINARY = [
    ("atan2", 1.0, 1.0, 0.7853982),
    ("pow", 2.0, 10.0, 1024.0),
    ("fmod", 7.0, 4.0, 3.0),
    ("hypot", 3.0, 4.0, 5.0),
]


@kernel
def unary_math(x: np.ndarray, out: np.ndarray):
    """Every unary intrinsic in one body.

    Each column of `x` holds the input for one function, because their domains differ: `asin`
    wants [-1, 1], `log` wants a positive argument, `cbrt` wants a cube.
    """
    for i in range(out.shape[0]):
        out[i, 0] = sqrt(x[i, 0])
        out[i, 1] = sin(x[i, 1])
        out[i, 2] = cos(x[i, 2])
        out[i, 3] = tan(x[i, 3])
        out[i, 4] = asin(x[i, 4])
        out[i, 5] = acos(x[i, 5])
        out[i, 6] = atan(x[i, 6])
        out[i, 7] = sinh(x[i, 7])
        out[i, 8] = cosh(x[i, 8])
        out[i, 9] = tanh(x[i, 9])
        out[i, 10] = exp(x[i, 10])
        out[i, 11] = log(x[i, 11])
        out[i, 12] = log2(x[i, 12])
        out[i, 13] = log10(x[i, 13])
        out[i, 14] = floor(x[i, 14])
        out[i, 15] = ceil(x[i, 15])
        out[i, 16] = trunc(x[i, 16])
        out[i, 17] = fabs(x[i, 17])
        out[i, 18] = cbrt(x[i, 18])


@kernel
def binary_math(x: np.ndarray, y: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = atan2(x[i], y[i])
        out[i, 1] = pow(x[i], y[i])
        out[i, 2] = fmod(x[i], y[i])
        out[i, 3] = hypot(x[i], y[i])


def test_every_unary_intrinsic_agrees_between_the_host_and_the_device(runner):
    """One launch covers all of them: column k of `x` is function k's own input."""
    inputs = np.array([[value for _, value, _ in UNARY]], dtype=np.float32)
    out = np.zeros((1, len(UNARY)), dtype=np.float32)
    runner(unary_math, inputs, out)
    for column, (name, _, expected) in enumerate(UNARY):
        got = float(out[0, column])
        assert got == pytest.approx(expected, abs=1e-3), f"{name} -> {got}"


def test_every_binary_intrinsic(runner):
    for name, left, right, expected in BINARY:
        out = np.zeros((1, 4), dtype=np.float32)
        runner(binary_math, np.array([left], dtype=np.float32),
               np.array([right], dtype=np.float32), out)
        column = [entry[0] for entry in BINARY].index(name)
        got = float(out[0, column])
        assert got == pytest.approx(expected, abs=1e-3), f"{name} -> {got}"


def test_each_math_name_is_pygfs_object():
    """One implementation per name: the host body is pygf's, so the two cannot drift."""
    import pygf.funcs as pygf_funcs

    for name, _, _ in UNARY + [(n, 0.0, 0.0) for n, _, _, _ in BINARY]:
        assert getattr(__import__("pygpgpu"), name) is getattr(pygf_funcs, name)


def test_a_math_name_is_recognized_as_a_builtin_not_a_helper():
    """The regression: `sqrt` was not in the builtin set, so a kernel body that called it was
    translated as a device helper and pygf's own body failed to compile."""
    from pygpgpu.kernel.builtin_registry import builtin_name_of

    assert builtin_name_of(sqrt) == "sqrt"
    assert builtin_name_of(pow) == "pow"
    assert builtin_name_of(sin) == "sin"


def test_the_math_names_appear_in_every_table():
    """`builtins.__all__`, the registry and the device table have to agree."""
    from pygpgpu.kernel import builtins as builtin_namespace
    from pygpgpu.kernel import device_builtins
    from pygpgpu.kernel.builtin_registry import BUILTIN_FUNCTIONS

    for name in [entry[0] for entry in UNARY] + [entry[0] for entry in BINARY]:
        assert name in builtin_namespace.__all__
        assert name in BUILTIN_FUNCTIONS
        assert name in device_builtins.GENERIC_BUILTINS
