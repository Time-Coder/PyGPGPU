"""The builtins tests, split out of `test_kernel_complex.py`.

The subject is builtins.
"""
from __future__ import annotations

from _helpers.test_kernel_complex import complex_builtins as _BUILTINS
from _helpers.test_kernel_complex import (
    DOUBLE,
    FLOAT,
    _predicate_mask,
    builtin_clamp_saturate,
    builtin_fmin_fmax,
    builtin_min_max,
    builtin_mix_mad,
    builtin_rint,
    builtin_sign,
    builtin_transcendental,
    numpy_clip,
    numpy_minimum_maximum,
    numpy_mix_mad,
    numpy_predicates,
    numpy_rint,
    numpy_round,
    numpy_sign,
    refuses_floor,
    refuses_fract,
    refuses_hypot,
    refuses_round,
    refuses_rsqrt,
    refuses_step,
)

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_builtin_transcendentals_answer_complex(runner, dtype):
    """`sqrt(z)` and its siblings are pygf's functions, so the device calls the header."""
    z = dtype(1 + 2j)
    out = np.zeros(2, dtype=dtype)
    runner(builtin_transcendental, z, out)
    expected = (np.sqrt(z) + np.exp(z) + np.log(z) + np.sin(z) + np.exp2(z)).astype(dtype)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_sign_answers_the_unit_direction(runner, dtype):
    """numpy's complex `sign`, under both spellings -- and zero at zero, not a NaN."""
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1 + 2j), dtype(-3 + 0.5j), dtype(0)):
        runner(builtin_sign, value, out)
        np.testing.assert_allclose(out[0], np.sign(value), rtol=1e-6, atol=1e-7)
        runner(numpy_sign, value, out)
        np.testing.assert_allclose(out[0], np.sign(value), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_rounding_a_complex_value_rounds_each_part(runner, dtype):
    """`np.round(z)`, `np.rint(z)` and this package's `rint(z)` all round both parts to even.

    `rint` is its own host body (`sci_math`), so it answers a complex value by rounding each
    part with python's `round` -- the same rule the device's `c64_rint` implements. Python's
    *builtin* `round(z)` still refuses one, which is a different name.
    """
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1.5 + 2.5j), dtype(2.5 - 3.5j), dtype(-1.5 + 0.5j)):
        runner(numpy_round, value, out)
        np.testing.assert_allclose(out[0], np.round(value))
        runner(numpy_rint, value, out)
        np.testing.assert_allclose(out[0], np.rint(value))
        runner(builtin_rint, value, out)
        np.testing.assert_allclose(out[0], np.rint(value))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_fmin_and_fmax_ignore_a_nan_operand(runner, dtype):
    """`fmin`/`fmax` are the NaN-ignoring pair, and numpy has two families for a reason.

    `min`/`max` (and `np.minimum`/`np.maximum`) propagate a NaN, because it has a defined
    place in their order; `fmin`/`fmax` return the operand that is *not* NaN. A complex
    value is NaN when either part is, so the device needs the second function rather than
    the ordering one it used to emit.
    """
    normal = dtype(3 - 1j)
    nan_real = dtype(complex(float("nan"), 1.0))
    nan_imag = dtype(complex(1.0, float("nan")))
    out = np.zeros(1, dtype=dtype)
    for pair in ((normal, normal), (nan_real, normal), (normal, nan_real),
                 (nan_imag, normal), (normal, nan_imag)):
        out[0] = 0
        runner(builtin_fmin_fmax, pair[0], pair[1], out)
        expected = np.fmin(pair[0], pair[1]) + np.fmax(pair[0], pair[1])
        np.testing.assert_allclose(out[0], expected)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_minimum_and_maximum_are_lexicographic(runner, dtype):
    """The real part decides and the imaginary one breaks the tie, as numpy does it."""
    z = dtype(1 + 2j)
    out = np.zeros(1, dtype=dtype)
    for w in (dtype(3 - 1j), dtype(1 - 1j), dtype(3 + 0j)):
        runner(builtin_min_max, z, w, out)
        np.testing.assert_allclose(out[0], np.minimum(z, w) + np.maximum(z, w))
        runner(numpy_minimum_maximum, z, w, out)
        np.testing.assert_allclose(out[0], np.minimum(z, w) + np.maximum(z, w))


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_clamp_and_clip_are_the_lexicographic_pair(runner, dtype):
    """`clamp(z, 0, 1)`, `saturate(z)` and `np.clip(z, 0, 1)` are one operation."""
    out = np.zeros(1, dtype=dtype)
    for value in (dtype(1 + 2j), dtype(0.5 + 0.25j), dtype(4 - 3j), dtype(-1 + 0j)):
        runner(builtin_clamp_saturate, value, out)
        np.testing.assert_allclose(out[0], np.clip(value, 0, 1) * 2)
        runner(numpy_clip, value, out)
        np.testing.assert_allclose(out[0], np.clip(value, 0, 1) * 2)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_mix_and_mad_are_the_complex_arithmetic(runner, dtype):
    """`mix(z, w, 0.5)` blends and `mad(z, w, v)` is a product plus a sum."""
    z, w, v = dtype(1 + 2j), dtype(3 - 1j), dtype(0.5 + 0.5j)
    out = np.zeros(1, dtype=dtype)
    runner(builtin_mix_mad, z, w, v, out)
    runner(numpy_mix_mad, z, w, v, out)
    expected = (z * (1 - 0.5) + w * 0.5) + (z * w + v)
    np.testing.assert_allclose(out[0], expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [FLOAT, DOUBLE], ids=["complex64", "complex128"])
def test_the_predicates_answer_for_either_part(runner, dtype):
    """`np.isnan`/`isinf`/`isfinite` are defined on a complex value in numpy."""
    out = np.zeros(1, dtype=np.float32)
    for value in (complex(np.nan, 1.0), complex(1.0, np.inf), dtype(1 + 2j),
                  complex(np.inf, np.nan)):
        runner(numpy_predicates, dtype(value), out)
        assert int(out[0]) == _predicate_mask(value), value


@pytest.mark.parametrize("function, expected", [
    (refuses_round, "use `np.round(z)`"),
    (refuses_rsqrt, "use `1 / np.sqrt(z)`"),
    (refuses_fract, "np.floor"),
    (refuses_step, "no ordering"),
    (refuses_floor, "no ordering"),
    (refuses_hypot, "np.abs(a - b)"),
], ids=["round", "rsqrt", "fract", "step", "floor", "hypot"])
def test_a_real_only_builtin_is_refused_for_a_complex_value(function, expected):
    """The refusal names the operation to write instead of a compiler's type error."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    complex_scalar = infer_ctype(1 + 2j)
    complex_array = infer_ctype(np.zeros(2, np.complex64))
    parameters: int = len(function.raw_func.__code__.co_varnames[:function.raw_func.__code__.co_argcount])
    types = [complex_scalar] * (parameters - 1) + [complex_array]
    translator = FunctionTranslator(function.raw_func, types, "cuda",
                                    lambda name: getattr(_BUILTINS, name, None), {}, is_kernel=True)
    with pytest.raises(SyntaxError) as caught:
        translator.translate()
    assert expected in str(caught.value)
    assert "complex" in str(caught.value)
