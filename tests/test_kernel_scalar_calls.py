"""Python builtins, `math` and `numpy` maths on a scalar, on every backend.

A kernel body may write `abs(v)`, `math.sqrt(v)` or `np.sqrt(v)`, and the three spellings have
to agree with what python does on one element — the device has a form of each, so the only
question is whether the translator knows it.

Two regressions are pinned here:

* `math.floor`/`ceil`/`trunc` answered the wrong number. They return an `int` in python, and
  the argument was being cast to the result type, so the device saw `floor((int)x)` — the
  integer overload, or no overload at all. `ceil(2.1)` came back 2. Only the *result* may be
  cast; the argument stays a float.
* `math.isnan`/`isinf`/`isfinite` failed the same way on CUDA ("more than one instance of
  overloaded function"): the predicate takes a float, not the int its answer is.

One known limit is recorded at the bottom: `math.trunc` cannot work on the interpreter backend,
because numpy's `float32` has no `__trunc__` and the interpreter runs the call unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def builtin_calls(flag: np.ndarray, out: np.ndarray):
    """The python builtins that have a device form."""
    for i in range(out.shape[0]):
        out[i, 0] = abs(-3.5)
        out[i, 1] = min(3.5, 2.0)
        out[i, 2] = max(3.5, 2.0)
        out[i, 3] = round(2.6)
        out[i, 4] = pow(3.0, 2.0)
        out[i, 5] = float(3.0)
        out[i, 6] = int(3.7)
        out[i, 7] = len(flag)


@kernel
def math_unary(flag: np.ndarray, out: np.ndarray):
    """One column per unary `math` function; the input is a constant so the answer is known."""
    for i in range(out.shape[0]):
        out[i, 0] = math.sqrt(4.0)
        out[i, 1] = math.sin(0.0)
        out[i, 2] = math.cos(0.0)
        out[i, 3] = math.tan(0.0)
        out[i, 4] = math.asin(0.5)
        out[i, 5] = math.acos(0.5)
        out[i, 6] = math.atan(1.0)
        out[i, 7] = math.sinh(0.0)
        out[i, 8] = math.cosh(0.0)
        out[i, 9] = math.tanh(0.0)
        out[i, 10] = math.exp(0.0)
        out[i, 11] = math.log(1.0)
        out[i, 12] = math.log2(8.0)
        out[i, 13] = math.log10(100.0)
        out[i, 14] = math.log1p(0.0)
        out[i, 15] = math.expm1(0.0)
        out[i, 16] = math.fabs(-2.5)
        out[i, 17] = math.floor(2.7)
        out[i, 18] = math.ceil(2.1)
        out[i, 19] = math.cbrt(8.0)
        out[i, 20] = math.degrees(3.14159265)
        out[i, 21] = math.radians(180.0)


@kernel
def math_binary(flag: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = math.atan2(1.0, 1.0)
        out[i, 1] = math.pow(2.0, 10.0)
        out[i, 2] = math.fmod(7.0, 4.0)
        out[i, 3] = math.hypot(3.0, 4.0)
        out[i, 4] = math.copysign(-3.0, 2.0)
        out[i, 5] = math.ldexp(3.0, 2)


@kernel
def math_predicates(x: np.ndarray, out: np.ndarray):
    """One row per input value: finiteness, NaN-ness and infinity-ness."""
    for i in range(out.shape[0]):
        out[i, 0] = 1.0 if math.isfinite(x[i]) else 0.0
        out[i, 1] = 1.0 if math.isnan(x[i]) else 0.0
        out[i, 2] = 1.0 if math.isinf(x[i]) else 0.0


@kernel
def numpy_unary(flag: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = np.sqrt(4.0)
        out[i, 1] = np.sin(0.0)
        out[i, 2] = np.cos(0.0)
        out[i, 3] = np.tanh(0.0)
        out[i, 4] = np.exp(0.0)
        out[i, 5] = np.log(1.0)
        out[i, 6] = np.log2(8.0)
        out[i, 7] = np.abs(-2.5)
        out[i, 8] = np.fabs(-2.5)
        out[i, 9] = np.floor(2.7)
        out[i, 10] = np.ceil(2.1)
        out[i, 11] = np.trunc(-2.7)
        out[i, 12] = np.round(2.6)
        out[i, 13] = np.square(3.0)
        out[i, 14] = np.reciprocal(4.0)


@kernel
def numpy_binary(flag: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i, 0] = np.power(2.0, 10.0)
        out[i, 1] = np.hypot(3.0, 4.0)
        out[i, 2] = np.arctan2(1.0, 1.0)
        out[i, 3] = np.minimum(3.5, 2.0)
        out[i, 4] = np.maximum(3.5, 2.0)


FLAG = np.zeros(1, dtype=np.float32)

BUILTINS = [(0, 3.5), (1, 2.0), (2, 3.5), (3, 3.0), (4, 9.0), (5, 3.0), (6, 3.0), (7, 1.0)]

MATH_UNARY = [
    (0, 2.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, 0.5235988), (5, 1.0471976),
    (6, 0.7853982), (7, 0.0), (8, 1.0), (9, 0.0), (10, 1.0), (11, 0.0),
    (12, 3.0), (13, 2.0), (14, 0.0), (15, 0.0), (16, 2.5),
    (17, 2.0), (18, 3.0), (19, 2.0), (20, 180.0), (21, 3.1415927),
]

MATH_BINARY = [(0, 0.7853982), (1, 1024.0), (2, 3.0), (3, 5.0), (4, 3.0), (5, 12.0)]

NUMPY_UNARY = [
    (0, 2.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, 1.0), (5, 0.0), (6, 3.0),
    (7, 2.5), (8, 2.5), (9, 2.0), (10, 3.0), (11, -2.0), (12, 3.0),
    (13, 9.0), (14, 0.25),
]

NUMPY_BINARY = [(0, 1024.0), (1, 5.0), (2, 0.7853982), (3, 2.0), (4, 3.5)]

GROUPS = [
    ("builtins", builtin_calls, BUILTINS),
    ("math unary", math_unary, MATH_UNARY),
    ("math binary", math_binary, MATH_BINARY),
    ("numpy unary", numpy_unary, NUMPY_UNARY),
    ("numpy binary", numpy_binary, NUMPY_BINARY),
]


@pytest.mark.parametrize("label, function, expected", GROUPS,
                         ids=[group[0] for group in GROUPS])
def test_scalar_calls(runner, label, function, expected):
    width = max(column for column, _ in expected) + 1
    out = np.zeros((1, width), dtype=np.float32)
    runner(function, FLAG, out)
    for column, value in expected:
        assert float(out[0, column]) == pytest.approx(value, abs=1e-3), (
            f"{label} column {column}")


@pytest.mark.parametrize("value, expected", [(2.1, 3.0), (2.9, 3.0), (-2.1, -2.0)])
def test_ceil_is_not_applied_to_a_truncated_argument(runner, value, expected):
    """The regression: `ceil((int)x)` answered `ceil(2)` = 2 for an input of 2.1."""
    out = np.zeros((1, 2), dtype=np.float32)
    runner(rounding_kernel, np.array([value], dtype=np.float32), out)
    assert float(out[0, 0]) == pytest.approx(expected)


@pytest.mark.parametrize("value, expected", [(2.7, 2.0), (2.9, 2.0), (-2.1, -3.0)])
def test_floor(runner, value, expected):
    out = np.zeros((1, 2), dtype=np.float32)
    runner(rounding_kernel, np.array([value], dtype=np.float32), out)
    assert float(out[0, 1]) == pytest.approx(expected)


@kernel
def rounding_kernel(x: np.ndarray, out: np.ndarray):
    """`ceil` in column 0, `floor` in column 1, so one launch checks both."""
    for i in range(out.shape[0]):
        out[i, 0] = math.ceil(x[i])
        out[i, 1] = math.floor(x[i])


PREDICATE_INPUTS = [1.0, float("nan"), float("inf"), float("-inf")]
PREDICATE_EXPECTED = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
]


def test_math_predicates(runner):
    """isnan/isinf/isfinite used to cast their argument to the int they answer."""
    out = np.zeros((len(PREDICATE_INPUTS), 3), dtype=np.float32)
    runner(math_predicates, np.array(PREDICATE_INPUTS, dtype=np.float32), out)
    for row, expected in enumerate(PREDICATE_EXPECTED):
        for column, value in enumerate(expected):
            assert float(out[row, column]) == pytest.approx(value), (row, column)


def test_math_trunc_cannot_run_on_the_interpreter():
    """Recorded, not desired: the interpreter executes the body unchanged, so `math.trunc`
    receives the numpy scalar that `x[i]` is, and numpy's float32 has no `__trunc__`. The
    device backends are fine, and `np.trunc(x[i])` or pygf's `trunc` work everywhere."""
    with pytest.raises(TypeError):
        math.trunc(np.float32(2.7))


def mix32(tid, seed, mask):
    """The shift-and-multiply mix a random number generator uses.

    It only means what it is supposed to under **unsigned** arithmetic: a signed 32-bit
    multiply overflows and `x >> 16` fills with ones, and the result comes back negative.
    """
    x = tid & mask
    x ^= (x >> 16)
    x = (x * 0x9E3779B9) & mask
    return x / 4294967295.0


@kernel
def unsigned_mix(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = mix32(np.uint32(i + 1), np.uint32(12345), np.uint32(0xFFFFFFFF))


@kernel
def signed_mix(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = mix32(i + 1, 12345, 0xFFFFFFFF)


def test_a_numpy_scalar_keeps_its_width_and_signedness(runner, gpu_backend):
    """`np.uint32(x)` reaches the device as `unsigned`, so the mix stays unsigned.

    A python `int` has no width of its own, and collapsing every integer to `int` made the
    unsigned mix signed: `x * 0x9E3779B9` overflowed and `x >> 16` filled with ones, so the
    generator answered a negative number where python answers 0.618.
    """
    from pygpgpu.kernel.translator import DTYPE_MAP, infer_ctype

    assert infer_ctype(np.uint32(5)) == DTYPE_MAP["uint32"]
    assert infer_ctype(np.uint64(5)) == DTYPE_MAP["uint64"]
    assert infer_ctype(np.int64(5)) == DTYPE_MAP["int64"]
    # The narrow widths keep the answer they always had (see
    # tests/test_kernel_narrow_arguments.py): an 8-bit value is an int, a float64 a float.
    assert infer_ctype(np.uint8(5)) == "int"
    assert infer_ctype(np.float64(5.0)) == "float"

    out = np.zeros(1, dtype=np.float32)
    runner(unsigned_mix, out)
    assert float(out[0]) == pytest.approx(0.618034, abs=1e-5)
    # The signed spelling gives a different (wrong) answer, which is what the width fixes.
    # The interpreter backend runs the body unchanged and keeps python's own answer, so
    # the comparison only holds where the code is actually translated.
    if gpu_backend is not None:
        runner(signed_mix, out)
        assert float(out[0]) < 0.0
