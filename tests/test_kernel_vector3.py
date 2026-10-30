"""Three-component vectors: construction, colour component names, swizzles.

A 3-component vector is padded to 16 bytes for the device ABI, so its ctypes struct carries a
trailing `_` field. Everything that counts components has to ignore it -- the constructor
used to require four arguments for `float3(a, b, c)`, and the print layout used to render
the padding as a fourth component.

Colour-style names (`color.r`) are normalised to the device spelling (`color.x`): OpenCL C
accepts both, CUDA's vector types only have x/y/z/w.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import color3f, float3, float4

from pygpgpu import kernel


@kernel
def build_and_read(out: np.ndarray):
    """Construct a float3 from three scalars and read it back by component."""
    for i in range(out.shape[0]):
        v = float3(1.0, 2.0, 3.0)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def colour_components(out: np.ndarray):
    """Read a colour through its r/g/b names, the way the raytracer helper does."""
    for i in range(out.shape[0]):
        c = float3(0.25, 0.5, 1.0)
        out[i] = c.r + 10.0 * c.g + 100.0 * c.b


@kernel
def colour_swizzle(out: np.ndarray):
    """A colour swizzle builds the matching vector: `.rgb` of a float4 is a float3."""
    for i in range(out.shape[0]):
        c = float4(0.25, 0.5, 1.0, 0.125)
        rgb = c.rgb
        out[i] = rgb.x + 10.0 * rgb.y + 100.0 * rgb.z


@kernel
def alpha_of_a_four_vector(out: np.ndarray):
    for i in range(out.shape[0]):
        c = float4(1.0, 2.0, 3.0, 4.0)
        out[i] = c.a


@kernel
def three_from_values(x: np.ndarray, out: np.ndarray):
    """A float3 built from runtime values, then summed back out."""
    for i in range(out.shape[0]):
        v = float3(x[i], x[i] * 2.0, x[i] * 3.0)
        out[i] = v.x + v.y + v.z


def test_a_three_component_constructor_takes_three_arguments(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(build_and_read, out)
    assert float(out[0]) == 321.0


def test_colour_component_names_read_the_same_components(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(colour_components, out)
    assert float(out[0]) == pytest.approx(0.25 + 5.0 + 100.0)


def test_a_colour_swizzle_yields_the_matching_vector(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(colour_swizzle, out)
    assert float(out[0]) == pytest.approx(0.25 + 5.0 + 100.0)


def test_alpha_is_the_fourth_component(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(alpha_of_a_four_vector, out)
    assert float(out[0]) == 4.0


def test_a_three_component_vector_from_runtime_values(runner):
    x = np.array([2.0], dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    runner(three_from_values, x, out)
    assert float(out[0]) == pytest.approx(12.0)


def test_the_host_object_agrees_with_the_device_vector():
    """`pygf`'s float3 is the host side: same three components, same colour names."""
    v = float3(1.0, 2.0, 3.0)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    assert (v.r, v.g, v.b) == (1.0, 2.0, 3.0)
    assert isinstance(color3f(1.0, 2.0, 3.0).rgb, float3)


@kernel
def wrong_arity(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float3(1.0, 2.0, 3.0, 4.0).x


def test_four_arguments_for_a_three_vector_is_refused(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(wrong_arity, np.zeros(1, dtype=np.float32))
    assert "3" in str(caught.value)


# ---------------------------------------------------------------------------
# The argument forms `pygf` accepts: none (zeros), one (broadcast), or one each
# ---------------------------------------------------------------------------


@kernel
def default_constructed(out: np.ndarray):
    """`float3()` is all zeros on the host, so it must be on the device too."""
    for i in range(out.shape[0]):
        v = float3()
        out[i] = v.x + v.y + v.z


@kernel
def broadcast_constructed(out: np.ndarray):
    """`float3(k)` fills every component with k, as `pygf` does."""
    for i in range(out.shape[0]):
        v = float3(2.5)
        out[i] = v.x + 10.0 * v.y + 100.0 * v.z


@kernel
def default_four_vector(out: np.ndarray):
    for i in range(out.shape[0]):
        v = float4()
        out[i] = v.x + v.y + v.z + v.w


def test_a_default_constructor_is_all_zeros(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(default_constructed, out)
    assert float(out[0]) == 0.0


def test_one_argument_broadcasts(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(broadcast_constructed, out)
    assert float(out[0]) == pytest.approx(2.5 * 111.0)


def test_a_four_component_default_constructor(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(default_four_vector, out)
    assert float(out[0]) == 0.0


def test_the_host_constructor_agrees_on_the_forms():
    assert float3() == float3(0.0, 0.0, 0.0)
    assert float3(2.5) == float3(2.5, 2.5, 2.5)


def test_two_arguments_are_refused_like_python():
    """`pygf` rejects a partial list, so the device path must too."""
    with pytest.raises(ValueError):
        float4(1.0, 2.0)



@kernel
def missing_component(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = float3(1.0, 2.0, 3.0).a


def test_a_component_a_three_vector_does_not_have_is_refused(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(missing_component, np.zeros(1, dtype=np.float32))
    assert "no component" in str(caught.value)
