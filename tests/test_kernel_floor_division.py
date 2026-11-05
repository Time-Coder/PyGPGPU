"""Integer `//` and `%` follow python's floor semantics, not C's truncation.

The two disagree as soon as an operand is negative:

    -7 // 2  ->  -4   (python floors)      -3   (C truncates toward zero)
    -7 % 2   ->   1   (python: divisor's sign)   -1   (C: dividend's sign)

A kernel that divides or takes a remainder on a value that may be negative therefore used
to answer differently from the same function run in python -- and silently, because both
results are plausible-looking numbers. The translator emits a small device helper that
applies the one-step correction, so the two sides agree for every sign.

The unsigned widths need no correction (their remainder is never negative), and the plain
C operator is emitted there; that is checked too, so the helper is not introduced where it
would only be dead code.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@kernel
def floor_div_two(values: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = values[i] // 2


@kernel
def floor_mod_two(values: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = values[i] % 2


@kernel
def floor_div_three(values: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = values[i] // 3


@kernel
def floor_mod_three(values: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = values[i] % 3


@kernel
def floor_div_augmented(values: np.ndarray, out: np.ndarray):
    """`//=` and `%=`, which carry the same correction as the binary form."""
    for i in range(out.shape[0]):
        a = values[i]
        a //= 2
        b = values[i]
        b %= 2
        out[i] = a * 10 + b


# A sign-changing mix: both directions, an exact multiple, and zero.
SIGNED = np.array([7, -7, 5, -5, 0, -1, -8, 8, -9, 9], dtype=np.int32)
SIGNED64 = np.array([7, -7, 8, -8, 0, -1, -9, 9], dtype=np.int64)
UNSIGNED = np.array([7, 9, 0, 5, 11, 3], dtype=np.uint32)


# ---------------------------------------------------------------------------
# Signed widths: the correction is what makes these agree
# ---------------------------------------------------------------------------


def test_floor_division_matches_python_for_negative_operands(runner):
    """`-7 // 2` is -4, not C's -3."""
    out = np.zeros(len(SIGNED), dtype=np.int32)
    runner(floor_div_two, SIGNED, out)
    np.testing.assert_array_equal(out, np.array([int(v // 2) for v in SIGNED]))


def test_floor_modulo_matches_python_for_negative_operands(runner):
    """`-7 % 2` is 1: python's remainder carries the *divisor's* sign."""
    out = np.zeros(len(SIGNED), dtype=np.int32)
    runner(floor_mod_two, SIGNED, out)
    np.testing.assert_array_equal(out, np.array([int(v % 2) for v in SIGNED]))


def test_the_correction_is_not_only_for_two(runner):
    """A divisor other than 2 exercises the same step, with different quotients."""
    out_div = np.zeros(len(SIGNED), dtype=np.int32)
    out_mod = np.zeros(len(SIGNED), dtype=np.int32)
    runner(floor_div_three, SIGNED, out_div)
    runner(floor_mod_three, SIGNED, out_mod)
    np.testing.assert_array_equal(out_div, np.array([int(v // 3) for v in SIGNED]))
    np.testing.assert_array_equal(out_mod, np.array([int(v % 3) for v in SIGNED]))


def test_a_64_bit_width_needs_the_same_correction(runner):
    """The helper is specialized per C type, so `long long` must agree as well."""
    out_div = np.zeros(len(SIGNED64), dtype=np.int64)
    out_mod = np.zeros(len(SIGNED64), dtype=np.int64)
    runner(floor_div_three, SIGNED64, out_div)
    runner(floor_mod_three, SIGNED64, out_mod)
    np.testing.assert_array_equal(out_div, np.array([int(v // 3) for v in SIGNED64]))
    np.testing.assert_array_equal(out_mod, np.array([int(v % 3) for v in SIGNED64]))


def test_the_augmented_forms_floor_too(runner):
    """`//=` used to be a plain `/=`, which truncated."""
    out = np.zeros(len(SIGNED), dtype=np.int32)
    runner(floor_div_augmented, SIGNED, out)
    expected = []
    for value in SIGNED:
        quotient = int(value) // 2
        remainder = int(value) % 2
        expected.append(quotient * 10 + remainder)
    np.testing.assert_array_equal(out, np.array(expected))


# ---------------------------------------------------------------------------
# Unsigned widths: no correction is emitted, and none is needed
# ---------------------------------------------------------------------------


def test_an_unsigned_width_agrees_without_a_helper(runner):
    """A uint's remainder is never negative, so C's own operator already matches."""
    out_div = np.zeros(len(UNSIGNED), dtype=np.uint32)
    out_mod = np.zeros(len(UNSIGNED), dtype=np.uint32)
    runner(floor_div_three, UNSIGNED, out_div)
    runner(floor_mod_three, UNSIGNED, out_mod)
    np.testing.assert_array_equal(out_div, np.array([int(v // 3) for v in UNSIGNED],
                                                    dtype=np.uint32))
    np.testing.assert_array_equal(out_mod, np.array([int(v % 3) for v in UNSIGNED],
                                                    dtype=np.uint32))


# ---------------------------------------------------------------------------
# The generated source
# ---------------------------------------------------------------------------


def _source(func, backend: str, values_type=None) -> str:
    """The device source a kernel renders, with the helper definition included."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    raw = func.raw_func if hasattr(func, "raw_func") else func
    f = infer_ctype(np.zeros(4, dtype=np.float32))
    element = infer_ctype(np.zeros(4, dtype=np.int32)) if values_type is None else values_type
    translator = FunctionTranslator(raw, [element, f], backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    helpers = "\n".join(translator.device_helpers)
    kernels = "\n".join(piece for _, piece in translator.render_kernels("k"))
    return helpers + "\n" + kernels


@pytest.mark.parametrize("backend", ["opencl", "cuda", "hip"])
def test_the_helper_is_declared_and_called(backend):
    """A signed `//` calls the helper, and the helper's definition comes with it."""
    source = _source(floor_div_two, backend)
    assert "__pygpgpu_floor_div_int" in source
    # The definition, not only the call: C needs the callee before its caller.
    assert "static inline" in source
    assert source.index("static inline") < source.index("out[i]")


def test_the_helper_name_is_a_valid_c_identifier():
    """A multi-word C type cannot appear in a name: `long long` is joined, not spaced."""
    from pygpgpu.kernel.device_builtins import python_floor_name

    assert python_floor_name("floor_div", "long long") == "__pygpgpu_floor_div_longlong"
    assert " " not in python_floor_name("floor_mod", "unsigned int")


@kernel
def unsigned_floor_div(values: np.ndarray, out: np.ndarray):
    """A uint32 `//`: the plain C operator already matches python."""
    for i in range(out.shape[0]):
        out[i] = values[i] // 3


def test_an_unsigned_width_emits_the_plain_operator():
    """The helper is withheld where it would only be dead code."""
    from pygpgpu.kernel.decorator import infer_ctype

    source = _source(unsigned_floor_div, "cuda",
                     infer_ctype(np.zeros(4, dtype=np.uint32)))
    assert "__pygpgpu_floor_div" not in source, source
    assert "/ 3" in source, source
