"""GLSL's widening vector constructors: a narrower vector fills the first slots.

`float3(float2(1.0, 2.0), 3.0)` is legal GLSL (`vec3(vec2, float)`) and the host's
`pygf` accepts it -- construction flattens every argument's components into the slots
left to right -- but the device constructor only knew the three fixed forms, so a
shader port had to take the vector apart by hand. The device now fills the slots the
same way for every family and width the vector registry carries.

The forms, exactly as `pygf` answers them:

* no arguments -- all zeros; one scalar -- broadcast;
* scalars and vectors mixed -- components fill left to right, and the arguments must
  supply *exactly* the width (over- and under-supply are a ValueError there);
* one vector argument alone -- the copy form, where a *wider* vector is truncated to
  its first components (`float2(float4(1, 2, 3, 4))` is `float2(1, 2)`) while a
  narrower one is refused;
* a component of a foreign family is converted per slot the way a ctypes field store
  converts (`int3(float2(1.9, 2.7), 3)` truncates toward zero).
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.vec_types import (char2, char4, double2, float2, float3, float4, int2,
                               int3, uint2, uint3)


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@kernel
def widen_prefix(out: np.ndarray):
    """`float3(float2, float)` -- the form the shader writer expects."""
    for i in range(out.shape[0]):
        v = float3(float2(1.0, 2.0), 3.0)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def widen_two_scalars(out: np.ndarray):
    """`float4(float2, float, float)`: the vector need not come last."""
    for i in range(out.shape[0]):
        v = float4(float2(1.0, 2.0), 3.0, 4.0)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z + 1000.0 * v.w


@kernel
def widen_three(out: np.ndarray):
    """`float4(float3, float)`."""
    for i in range(out.shape[0]):
        v = float4(float3(1.0, 2.0, 3.0), 4.0)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z + 1000.0 * v.w


@kernel
def widen_scalar_first(out: np.ndarray):
    """GLSL lets the scalars come first too: `float3(1.0, vec2)`."""
    for i in range(out.shape[0]):
        v = float3(1.0, float2(2.0, 3.0))
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def widen_two_vectors(out: np.ndarray):
    """Two vectors together: `float4(vec2, vec2)` fills four slots."""
    for i in range(out.shape[0]):
        v = float4(float2(1.0, 2.0), float2(3.0, 4.0))
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z + 1000.0 * v.w


@kernel
def widen_across_families(out: np.ndarray):
    """A foreign family's components convert per slot, like a field store."""
    for i in range(out.shape[0]):
        v = float3(int2(1, 2), 3.0)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def widen_truncating_slots(out: np.ndarray):
    """`int3(float2(1.9, 2.7), 3)` truncates toward zero, the C field-store rule."""
    for i in range(out.shape[0]):
        v = int3(float2(1.9, 2.7), 3)
        out[i] = v.x + 10 * v.y + 100 * v.z


@kernel
def copy_same_width(out: np.ndarray):
    """`float3(float3(...))` copies: every component fills one slot."""
    for i in range(out.shape[0]):
        v = float3(float3(1.0, 2.0, 3.0))
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def truncate_single_wider(out: np.ndarray):
    """`float2(float4(...))` keeps the first two components, as `pygf` does."""
    for i in range(out.shape[0]):
        v = float2(float4(1.0, 2.0, 3.0, 4.0))
        out[i] = v.x + 10.0 * v.y


@kernel
def widen_from_runtime_values(out: np.ndarray):
    """Runtime values travel through the widening form the same way."""
    for i in range(out.shape[0]):
        x = i + 2.0
        v = float4(float2(x, x * 2.0), x * 3.0, x * 4.0)
        out[i] = v.x + v.y + v.z + v.w


@kernel
def widen_unsigned_family(out: np.ndarray):
    """The uint family constructs too -- whose cast CUDA spells `unsigned int`."""
    for i in range(out.shape[0]):
        v = uint3(uint2(1, 2), 3)
        out[i] = v.x + 10 * v.y + 100 * v.z


@kernel
def widen_narrow_family(out: np.ndarray):
    """`char4(char2, char, char)` -- the narrow families the host library carries."""
    for i in range(out.shape[0]):
        v = char4(char2(1, 2), 3, 4)
        out[i] = v.x + 10 * v.y + 100 * v.z + 1000 * v.w


@kernel
def widen_double_family(out: np.ndarray):
    for i in range(out.shape[0]):
        v = double2(int2(3, 4))
        out[i] = v.x + v.y


# ---------------------------------------------------------------------------
# The values
# ---------------------------------------------------------------------------


def test_a_narrower_vector_then_a_scalar(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_prefix, out)
    assert float(out[0]) == 321.0


def test_a_narrower_vector_then_two_scalars(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_two_scalars, out)
    assert float(out[0]) == 4321.0


def test_a_three_vector_then_a_scalar(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_three, out)
    assert float(out[0]) == 4321.0


def test_scalars_may_come_first(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_scalar_first, out)
    assert float(out[0]) == 321.0


def test_two_vectors_fill_the_slots_together(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_two_vectors, out)
    assert float(out[0]) == 4321.0


def test_a_foreign_family_converts_per_slot(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_across_families, out)
    assert float(out[0]) == 321.0


def test_int_slots_truncate_like_the_host(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(widen_truncating_slots, out)
    assert int(out[0]) == 321


def test_the_same_width_copies(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(copy_same_width, out)
    assert float(out[0]) == 321.0


def test_one_wider_vector_is_truncated_to_its_first_components(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(truncate_single_wider, out)
    assert float(out[0]) == 21.0


def test_runtime_values_travel_through_the_widening_form(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widen_from_runtime_values, out)
    assert float(out[0]) == pytest.approx(20.0)


def test_the_unsigned_family_constructs(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(widen_unsigned_family, out)
    assert int(out[0]) == 321


def test_the_narrow_family_constructs(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(widen_narrow_family, out)
    assert int(out[0]) == 4321


def test_the_double_family_constructs(runner):
    out = np.zeros(1, dtype=np.float64)
    runner(widen_double_family, out)
    assert float(out[0]) == 7.0


# ---------------------------------------------------------------------------
# Refusals: pygf's ValueError forms are refused at translation time
# ---------------------------------------------------------------------------


@kernel
def too_few_components(out: np.ndarray):
    for i in range(out.shape[0]):
        v = float3(float2(1.0, 2.0))
        out[i] = v.x


@kernel
def too_many_components(out: np.ndarray):
    for i in range(out.shape[0]):
        v = float3(float2(1.0, 2.0), 3.0, 4.0)
        out[i] = v.x


@kernel
def four_scalars(out: np.ndarray):
    for i in range(out.shape[0]):
        v = float3(1.0, 2.0, 3.0, 4.0)
        out[i] = v.x


@kernel
def truncated_with_extra_arguments(out: np.ndarray):
    """Truncation is only the single-vector form: extra args are refused."""
    for i in range(out.shape[0]):
        v = float2(float3(1.0, 2.0, 3.0), 9.0)
        out[i] = v.x


@pytest.mark.parametrize("refused", [too_few_components, too_many_components,
                                     four_scalars, truncated_with_extra_arguments])
def test_an_over_or_under_supply_is_refused(gpu_backend, runner, refused):
    with pytest.raises(SyntaxError) as caught:
        runner(refused, np.zeros(1, dtype=np.float32))
    assert "component" in str(caught.value)


# ---------------------------------------------------------------------------
# The host agreement: the device forms are the ones `pygf` accepts
# ---------------------------------------------------------------------------


def test_the_host_constructor_answers_the_same_forms():
    assert float3(float2(1.0, 2.0), 3.0) == float3(1.0, 2.0, 3.0)
    assert float4(float2(1.0, 2.0), 3.0, 4.0) == float4(1.0, 2.0, 3.0, 4.0)
    assert float3(1.0, float2(2.0, 3.0)) == float3(1.0, 2.0, 3.0)
    assert float2(float4(1.0, 2.0, 3.0, 4.0)) == float2(1.0, 2.0)
    assert int3(float2(1.9, 2.7), 3) == int3(1, 2, 3)
    assert char4(char2(1, 2), 3, 4) == char4(1, 2, 3, 4)


def test_the_host_refuses_the_same_mis_supplies():
    with pytest.raises(ValueError):
        float3(float2(1.0, 2.0))
    with pytest.raises(ValueError):
        float3(float2(1.0, 2.0), 3.0, 4.0)
    with pytest.raises(ValueError):
        float3(1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        float2(float3(1.0, 2.0, 3.0), 9.0)
