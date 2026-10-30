"""`pygf`'s alias classes and the loop/domain rank agreement.

`pygf` gives one device vector several host names: `color3f`, `normal3f`, `point3f`,
`vector3f` and `texCoord3f` all subclass `float3`, and the `*d` variants subclass the double
ones. Device code has no such aliases, so the translator walks the class's bases to find the
real vector type and emits that constructor -- `color3f(r, g, b)` becomes `float3(r, g, b)`.
The host side keeps running the alias itself, so the two agree.

The second half is the loop check: `np.ndindex(arr.shape)` on a rank-3 array with a two-name
loop used to iterate only two dimensions and leave the third constant, which is a silent
wrong answer rather than an error.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygf import (
    color3d,
    color3f,
    color4f,
    normal3f,
    point3f,
    texCoord2f,
    texCoord3f,
    vector3f,
)

from pygpgpu import kernel
from pygpgpu.kernel.translator import _vec_registry


# ---------------------------------------------------------------------------
# Alias classes
# ---------------------------------------------------------------------------


@kernel
def alias_sum(out: np.ndarray):
    """One call per alias family; every one of them is the same float3."""
    for i in range(out.shape[0]):
        c = color3f(0.25, 0.5, 1.0)
        n = normal3f(1.0, 0.0, 0.0)
        p = point3f(0.0, 2.0, 0.0)
        v = vector3f(0.0, 0.0, 3.0)
        out[i] = c.r + c.g + c.b + n.x + p.y + v.z


@kernel
def double_alias(out: np.ndarray):
    for i in range(out.shape[0]):
        c = color3d(1.0, 2.0, 4.0)
        out[i] = c.x + c.y + c.z


@kernel
def two_and_four_component_aliases(out: np.ndarray):
    for i in range(out.shape[0]):
        t = texCoord2f(0.5, 0.25)
        a = color4f(0.0, 0.0, 0.0, 0.5)
        out[i] = t.x + t.y + a.a


@kernel
def three_component_texcoord(out: np.ndarray):
    for i in range(out.shape[0]):
        t = texCoord3f(1.0, 2.0, 4.0)
        out[i] = t.x + t.y + t.z


def test_every_alias_is_the_same_device_vector(runner):
    out = np.zeros(1, dtype=np.float64)
    runner(alias_sum, out)
    assert float(out[0]) == pytest.approx(0.25 + 0.5 + 1.0 + 1.0 + 2.0 + 3.0)


def test_the_double_aliases_map_to_double_vectors(runner):
    out = np.zeros(1, dtype=np.float64)
    runner(double_alias, out)
    assert float(out[0]) == pytest.approx(7.0)


def test_two_and_four_component_aliases(runner):
    out = np.zeros(1, dtype=np.float64)
    runner(two_and_four_component_aliases, out)
    assert float(out[0]) == pytest.approx(0.5 + 0.25 + 0.5)


def test_a_three_component_texcoord(runner):
    out = np.zeros(1, dtype=np.float64)
    runner(three_component_texcoord, out)
    assert float(out[0]) == pytest.approx(7.0)


def test_the_host_alias_and_the_device_vector_agree():
    """`color3f` is a `float3` on the host too, component for component."""
    c = color3f(0.25, 0.5, 1.0)
    assert (c.r, c.g, c.b) == (0.25, 0.5, 1.0)
    assert isinstance(c, __import__("pygf").float3)


@pytest.mark.parametrize("name, expected", [
    ("color3f", "float3"), ("normal3f", "float3"), ("point3f", "float3"),
    ("vector3f", "float3"), ("texCoord3f", "float3"), ("texCoord2f", "float2"),
    ("color4f", "float4"), ("color3d", "double3"), ("point3d", "double3"),
])
def test_the_alias_resolves_to_its_base_vector(name, expected):
    import pygf.alias as alias_module

    cls = getattr(alias_module, name)
    base = next((b.__name__ for b in cls.__mro__ if b.__name__ in _vec_registry()), None)
    assert base == expected


# ---------------------------------------------------------------------------
# The loop must name one index per dimension
# ---------------------------------------------------------------------------


@kernel
def two_names_over_three_dimensions(image: np.ndarray, out: np.ndarray):
    for j, i in np.ndindex(image.shape):
        out[i, j] = 1.0


def test_a_short_unpacking_is_refused(gpu_backend, runner):
    """The channel axis would never vary, so the kernel would be silently wrong."""
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    out = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        runner(two_names_over_three_dimensions, image, out)
    message: str = str(caught.value)
    assert "unpacks 2 index names" in message
    assert "3 dimensions" in message
    assert "image.shape[0], image.shape[1]" in message


@kernel
def full_unpacking(image: np.ndarray, out: np.ndarray):
    for i, j, k in np.ndindex(image.shape):
        out[i, j, k] = image[i, j, k]


def test_the_matching_unpacking_still_runs(runner):
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    out = np.zeros((2, 2, 3), dtype=np.uint8)
    runner(full_unpacking, image, out)
    np.testing.assert_array_equal(out, image)


@kernel
def explicit_subset(image: np.ndarray, out: np.ndarray):
    """Naming the two dimensions explicitly is the supported way to iterate a subset."""
    for j, i in np.ndindex(image.shape[0], image.shape[1]):
        out[j, i] = 1.0


def test_an_explicit_subset_is_accepted(runner):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    out = np.zeros((2, 3), dtype=np.float32)
    runner(explicit_subset, image, out)
    np.testing.assert_array_equal(out, np.ones((2, 3), dtype=np.float32))
