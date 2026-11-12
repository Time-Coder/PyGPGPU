"""Vector `%` and `//` carry python's floored rule, per component.

The scalar spellings floor both operators through the `__pygpgpu_*` device helpers
(see `tests/test_kernel_floor_division.py`), but the vector dispatch refused `Mod`
and `FloorDiv` outright: a shader that writes `uv % tiling` or `p // cell` had to be
taken apart per component by hand. The vector form now applies the same rule per
component, calling the *same* emitted helpers the scalar path declares -- no second
implementation of the correction -- and a scalar operand broadcasts the way the
arithmetic family's does.

The reference for every value is numpy (`np.mod` / `np.floor_divide`), which is also
what the host's `pygf` vectors compute, so the `python` backend and every device
answer alike for negative operands:

    float2(-7, -7) % float2(2, 2)  ->  (1, 1)   not C's (-1, -1)
    float2(-7, -7) // float2(2, 2) ->  (-4, -4) not C's (-3, -3)
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.vec_types import float2, float3, int2, uint2


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@kernel
def mod_vec_vec(out: np.ndarray):
    """`a % b` with both signs in one vector, the shape a tiling shader writes."""
    for i in range(out.shape[0]):
        a = float2(24.0, -7.0)
        b = float2(5.0, 2.0)
        c = a % b
        out[i] = c.x + 10.0 * c.y


@kernel
def mod_vec_scalar(out: np.ndarray):
    """`a % 5.0`: the scalar is shared by every component."""
    for i in range(out.shape[0]):
        a = float2(24.0, -7.0)
        c = a % 5.0
        out[i] = c.x + 10.0 * c.y


@kernel
def mod_scalar_vec(out: np.ndarray):
    """`24.0 % a`: the vector on the right decides the shape."""
    for i in range(out.shape[0]):
        a = float2(5.0, 2.0)
        c = 24.0 % a
        out[i] = c.x + 10.0 * c.y


@kernel
def floordiv_vec_vec(out: np.ndarray):
    for i in range(out.shape[0]):
        a = float2(24.0, -7.0)
        b = float2(5.0, 2.0)
        c = a // b
        out[i] = c.x + 10.0 * c.y


@kernel
def floordiv_vec_scalar(out: np.ndarray):
    for i in range(out.shape[0]):
        a = float2(24.0, -7.0)
        c = a // 2.0
        out[i] = c.x + 10.0 * c.y


@kernel
def floordiv_scalar_vec(out: np.ndarray):
    for i in range(out.shape[0]):
        a = float2(5.0, 2.0)
        c = 24.0 // a
        out[i] = c.x + 10.0 * c.y


@kernel
def mod_int_vec(out: np.ndarray):
    """The integer widths need the same correction: `-7 % 3` is 2, not C's -1."""
    for i in range(out.shape[0]):
        a = int2(-7, -9)
        b = int2(3, 4)
        c = a % b
        out[i] = c.x + 10 * c.y


@kernel
def floordiv_int_vec(out: np.ndarray):
    for i in range(out.shape[0]):
        a = int2(-7, -9)
        b = int2(2, 4)
        c = a // b
        out[i] = c.x + 10 * c.y


@kernel
def unsigned_mod_and_floordiv(out: np.ndarray):
    """An unsigned width's own operator already matches python."""
    for i in range(out.shape[0]):
        a = uint2(7, 9)
        b = uint2(3, 4)
        out[i] = (a % b).x + 10 * (a // b).y


@kernel
def mod_augassigned(out: np.ndarray):
    """`v %= b` stores the same per-component remainder the binary form produces."""
    for i in range(out.shape[0]):
        v = float2(24.0, -7.0)
        v %= float2(5.0, 2.0)
        out[i] = v.x + 10.0 * v.y


@kernel
def floordiv_augassigned(out: np.ndarray):
    for i in range(out.shape[0]):
        v = int2(-7, -9)
        v //= int2(2, 4)
        out[i] = v.x + 10 * v.y


@kernel
def mod_int_vec_float_scalar(out: np.ndarray):
    """An int vector with a float scalar promotes to the float form, as `7 % 2.5` does."""
    for i in range(out.shape[0]):
        v = int2(-7, -9)
        c = v % 2.5
        out[i] = c.x + 10 * c.y


@kernel
def floordiv_int_vec_float_scalar(out: np.ndarray):
    for i in range(out.shape[0]):
        v = int2(-7, -9)
        c = v // 2.5
        out[i] = c.x + 10 * c.y


@kernel
def mod_nested_operand(out: np.ndarray):
    """A computed operand: the wrapper call evaluates it once, like `a + b` does."""
    for i in range(out.shape[0]):
        a = float2(24.0, -7.0)
        b = float2(1.0, 1.0)
        c = (a + b) % float2(5.0, 2.0)
        out[i] = c.x + 10.0 * c.y


# ---------------------------------------------------------------------------
# The values: numpy's floored answers, combined the way each kernel stores them
# ---------------------------------------------------------------------------


def _combine(pair) -> float:
    """The `x + 10 * y` encoding every kernel above writes to `out`."""
    return float(pair[0]) + 10.0 * float(pair[1])


def _expected(a, b, op) -> float:
    """numpy's per-component answer for one kernel's operands."""
    left = np.asarray(a)
    right = np.asarray(b)
    return _combine(op(left, right))


FLOAT_A = (24.0, -7.0)
FLOAT_B = (5.0, 2.0)


def test_mod_matches_numpy_for_both_signs(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(mod_vec_vec, out)
    assert out[0] == pytest.approx(_expected(FLOAT_A, FLOAT_B, np.mod))


def test_mod_matches_numpy_for_a_scalar_divisor(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(mod_vec_scalar, out)
    assert out[0] == pytest.approx(_expected(FLOAT_A, (5.0, 5.0), np.mod))


def test_mod_matches_numpy_for_a_scalar_dividend(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(mod_scalar_vec, out)
    assert out[0] == pytest.approx(_expected((24.0, 24.0), FLOAT_B, np.mod))


def test_floordiv_matches_numpy_for_both_signs(runner):
    """`-7 // 2` is -4, the floored answer, not C's -3."""
    out = np.zeros(1, dtype=np.float32)
    runner(floordiv_vec_vec, out)
    assert out[0] == pytest.approx(_expected(FLOAT_A, FLOAT_B, np.floor_divide))


def test_floordiv_matches_numpy_for_a_scalar_divisor(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(floordiv_vec_scalar, out)
    assert out[0] == pytest.approx(_expected(FLOAT_A, (2.0, 2.0), np.floor_divide))


def test_floordiv_matches_numpy_for_a_scalar_dividend(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(floordiv_scalar_vec, out)
    assert out[0] == pytest.approx(_expected((24.0, 24.0), FLOAT_B, np.floor_divide))


def test_int_mod_matches_numpy_for_negative_operands(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(mod_int_vec, out)
    assert out[0] == _expected((-7, -9), (3, 4), np.mod)


def test_int_floordiv_matches_numpy_for_negative_operands(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(floordiv_int_vec, out)
    assert out[0] == _expected((-7, -9), (2, 4), np.floor_divide)


def test_unsigned_width_agrees_without_a_correction(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(unsigned_mod_and_floordiv, out)
    remainder_x = int(np.mod(np.uint32(7), np.uint32(3)))
    quotient_y = int(np.floor_divide(np.uint32(9), np.uint32(4)))
    assert out[0] == remainder_x + 10 * quotient_y


def test_the_augmented_forms_floor_too(runner):
    out_mod = np.zeros(1, dtype=np.float32)
    out_div = np.zeros(1, dtype=np.int32)
    runner(mod_augassigned, out_mod)
    runner(floordiv_augassigned, out_div)
    assert out_mod[0] == pytest.approx(_expected(FLOAT_A, FLOAT_B, np.mod))
    assert out_div[0] == _expected((-7, -9), (2, 4), np.floor_divide)


def test_int_vector_with_a_float_scalar_takes_the_float_form(runner):
    """The promoted width decides, the way the scalar spellings promote."""
    out = np.zeros(1, dtype=np.int32)
    runner(mod_int_vec_float_scalar, out)
    assert out[0] == int(np.mod(-7, 2.5)) + 10 * int(np.mod(-9, 2.5))


def test_int_vector_floordiv_with_a_float_scalar_floors(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(floordiv_int_vec_float_scalar, out)
    assert out[0] == int(np.floor(-7 / 2.5)) + 10 * int(np.floor(-9 / 2.5))


def test_a_computed_operand_is_evaluated_once(runner):
    """`(a + b) % c` answers the value python computes for the same expression."""
    out = np.zeros(1, dtype=np.float32)
    runner(mod_nested_operand, out)
    assert out[0] == pytest.approx(_expected((25.0, -6.0), FLOAT_B, np.mod))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@kernel
def mismatched_vectors(out: np.ndarray):
    for i in range(out.shape[0]):
        a = float2(1.0, 2.0)
        b = float3(1.0, 2.0, 3.0)
        out[i] = (a % b).x


def test_mismatched_vector_operands_are_refused(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(mismatched_vectors, np.zeros(1, dtype=np.float32))
    assert "same type" in str(caught.value)


# ---------------------------------------------------------------------------
# The generated source: one wrapper per shape, the scalar helper reused
# ---------------------------------------------------------------------------


def _source(func, backend: str) -> str:
    """The device source one kernel renders, with the helper definitions included."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    raw = func.raw_func if hasattr(func, "raw_func") else func
    f = infer_ctype(np.zeros(4, dtype=np.float32))
    translator = FunctionTranslator(raw, [f], backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    helpers = "\n".join(translator.device_helpers)
    kernels = "\n".join(piece for _, piece in translator.render_kernels("k"))
    return helpers + "\n" + kernels


WRAPPER_DEF = "vec_floor_mod_float2_float2(float2 a, float2 b)"


@pytest.mark.parametrize("backend", ["opencl", "cuda", "hip", "metal"])
def test_the_wrapper_calls_the_scalar_floor_helper(backend):
    """The vector form declares no correction of its own: it calls the scalar one."""
    source = _source(mod_vec_vec, backend)
    assert WRAPPER_DEF in source
    # The scalar helper the scalar `%` emits is there too, and the wrapper's body
    # is a per-component call of it -- the correction lives in exactly one place.
    assert "float __pygpgpu_float_mod_float(float a, float b)" in source
    assert "__pygpgpu_float_mod_float(a.x, b.x)" in source
    assert "__pygpgpu_float_mod_float(a.y, b.y)" in source


@pytest.mark.parametrize("backend", ["opencl", "cuda", "hip", "metal"])
def test_the_wrapper_qualifier_and_maker_follow_the_backend(backend):
    source = _source(mod_vec_vec, backend)
    definition = source[source.index(WRAPPER_DEF) - 60:]
    if backend in ("cuda", "hip"):
        assert "__device__ static inline float2" in definition
        assert "make_float2(" in source
    else:
        assert "static inline float2" in definition
        assert "(float2)(" in source


@pytest.mark.parametrize("backend", ["opencl", "cuda", "hip", "metal"])
def test_the_kernel_calls_the_wrapper_once_per_operation(backend):
    """An operand expression appears once, in the call -- not once per component."""
    source = _source(mod_vec_vec, backend)
    kernel_body = source[source.index("void k("):]
    assert kernel_body.count("vec_floor_mod_float2_float2(a, b)") == 1


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_the_fmod_spelling_follows_the_backend(backend):
    """CUDA spells the float form `fmodf`; OpenCL C and Metal name plain `fmod`."""
    source = _source(mod_vec_vec, backend)
    if backend == "cuda":
        assert "fmodf(" in source
    else:
        assert "fmod(" in source
        assert "fmodf(" not in source


def test_the_floating_floordiv_is_an_inline_floor_per_component():
    """A floating `//` has no scalar helper; the wrapper spells the same `floor`."""
    source = _source(floordiv_vec_vec, "cuda")
    assert "vec_floor_div_float2_float2" in source
    assert "((float)floor((float)(a.x) / (float)(b.x)))" in source
    assert "((float)floor((float)(a.y) / (float)(b.y)))" in source


def test_an_unsigned_width_keeps_the_plain_operator():
    """No correction helper is declared for uint; the wrapper spells `%` itself."""
    source = _source(unsigned_mod_and_floordiv, "opencl")
    assert "__pygpgpu_floor_mod" not in source
    assert "__pygpgpu_floor_div" not in source
    assert "(a.x % b.x)" in source
    assert "(a.y / b.y)" in source
