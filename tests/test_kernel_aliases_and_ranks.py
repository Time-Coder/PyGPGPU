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

from pygpgpu import kernel, get_global_id
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


@kernel
def writes_every_channel(image: np.ndarray, out: np.ndarray):
    for y, x, c in np.ndindex(image.shape):
        out[y, x, c] = 1.0


def test_a_flat_bracket_launches_the_whole_domain(gpu_backend):
    """The launch rank is independent of the traversal rank: a flat grid covers every element.

    A 3-D loop launched as a one-dimensional grid used to write 3 of 60 elements -- the loop
    read one index per launch axis, and a 1-D grid holds the other two axes at 0. The kernel
    now folds the launch's ids into one flat work-item position and decomposes it over the
    loop's own domain, so the two ranks no longer have to match.
    """
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    writes_every_channel[gpu_backend][(4 * 5 * 3,)](image, out)
    np.testing.assert_array_equal(out, np.ones((4, 5, 3), dtype=np.float32))


def test_the_loop_derives_a_matching_work_size(gpu_backend):
    """Without a bracket every channel of the domain is covered."""
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    writes_every_channel[gpu_backend](image, out)
    np.testing.assert_array_equal(out, np.ones((4, 5, 3), dtype=np.float32))


def test_a_bracket_of_the_right_rank_in_the_right_order_runs(gpu_backend):
    """A matching bracket is honoured, and axes are fastest-first (the loop's reverse)."""
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    writes_every_channel[gpu_backend][((3, 5, 4), None)](image, out)
    np.testing.assert_array_equal(out, np.ones((4, 5, 3), dtype=np.float32))


def test_a_bracket_with_the_axes_reversed_still_covers_the_domain(gpu_backend):
    """A right-rank bracket in the other order is just another launch shape now.

    It used to be the second silent failure: each axis was guarded against another axis's
    bound and the launch stopped at 45 of 60 elements. Because the work-item position is
    flat and decomposed afterwards, the axis order of the bracket no longer matters.
    """
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    writes_every_channel[gpu_backend][((4, 5, 3), None)](image, out)
    np.testing.assert_array_equal(out, np.ones((4, 5, 3), dtype=np.float32))


def test_a_bracket_that_stops_short_is_refused(gpu_backend):
    """Coverage is what matters: a launch with too few work items cannot visit the domain."""
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError) as caught:
        writes_every_channel[gpu_backend][(30,)](image, out)
    message: str = str(caught.value)
    assert "launches 30 work items" in message, message
    assert "traverses" in message, message


def test_an_oversized_bracket_still_runs(gpu_backend):
    """A grid larger than the domain is wasteful, not wrong: the bounds guard skips the rest."""
    image = np.zeros((4, 5, 3), dtype=np.float32)
    out = np.zeros((4, 5, 3), dtype=np.float32)
    writes_every_channel[gpu_backend][((8, 8, 8), None)](image, out)
    np.testing.assert_array_equal(out, np.ones((4, 5, 3), dtype=np.float32))


@kernel
def writes_every_axis_four(a: np.ndarray, out: np.ndarray):
    """More traversal axes than a device grid has (three), so the launch flattens."""
    for i, j, k, l in np.ndindex(a.shape):
        out[i, j, k, l] = 1.0


def test_a_four_dimensional_traversal_runs(gpu_backend):
    """Four axes cannot be launched as a grid, so the launch is one flat axis."""
    shape = (2, 3, 4, 5)
    out = np.zeros(shape, dtype=np.float32)
    writes_every_axis_four[gpu_backend](np.zeros(shape, dtype=np.float32), out)
    np.testing.assert_array_equal(out, np.ones(shape, dtype=np.float32))


@kernel
def copies_six_axes(src: np.ndarray, dst: np.ndarray):
    """Rank six, with an index-dependent payload so a mis-mapping shows up as a value."""
    for a, b, c, d, e, f in np.ndindex(src.shape):
        dst[a, b, c, d, e, f] = src[a, b, c, d, e, f]


def test_an_array_of_any_rank_is_accepted(gpu_backend):
    """No rank limit survives: an array argument may have more axes than a grid has.

    The axes of an array argument were never capped -- `_shape_var` emits one shape
    parameter per dimension -- but the *traversal* was, so a rank-six array could not be
    iterated. The launch now flattens, and every element has to arrive at its own address.
    """
    shape = (2, 2, 2, 2, 2, 2)
    src = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    dst = np.zeros(shape, dtype=np.float32)
    copies_six_axes[gpu_backend](src, dst)
    np.testing.assert_array_equal(dst, src)


@kernel
def copies_six_axes_flat(src: np.ndarray, dst: np.ndarray):
    """The same rank-six traversal, launched explicitly with one flat axis."""
    for a, b, c, d, e, f in np.ndindex(src.shape):
        dst[a, b, c, d, e, f] = src[a, b, c, d, e, f]


def test_a_high_rank_traversal_takes_a_flat_bracket(gpu_backend):
    """A rank-six domain is 64 work items, and a flat bracket of that size is accepted."""
    shape = (2, 2, 2, 2, 2, 2)
    src = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    dst = np.zeros(shape, dtype=np.float32)
    copies_six_axes_flat[gpu_backend][(64,)](src, dst)
    np.testing.assert_array_equal(dst, src)


@kernel
def gid_style_over_a_bracket(out: np.ndarray):
    """A gid-style body takes its domain from the bracket, so a flat grid is correct."""
    out[get_global_id(0)] = 1.0


def test_a_gid_style_body_keeps_its_flat_bracket(gpu_backend):
    """The rank check must not touch a body that declares no magic loop."""
    out = np.zeros(64, dtype=np.float32)
    gid_style_over_a_bracket[gpu_backend][(64,)](out)
    np.testing.assert_array_equal(out, np.ones(64, dtype=np.float32))


@kernel(domain="src")
def work_size_over_four_axes(src: np.ndarray, dst: np.ndarray):
    """A gid-style body whose declared domain has more axes than a grid has."""
    idx = get_global_id(0)
    dst[idx, 0, 0, 0] = src[idx, 0, 0, 0]


def test_a_domain_of_more_than_three_axes_folds_to_a_flat_launch(gpu_backend):
    """A grid has three axes, so a wider domain launches as `prod(shape)` work items.

    This is what the original auto-detection did (`math.prod(max_shape)` when the rank
    exceeded three), and it is legal: the fold changes the launch, not the domain. A
    gid-style body reads the flat position directly, so `get_global_id(0)` walks the whole
    domain and the axes it does not name read 0 -- which is why a body that wants the
    original axes spelled out iterates with `np.ndindex` instead.
    """
    shape = (2, 2, 3, 4)
    src = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    dst = np.zeros(shape, dtype=np.float32)
    work_size_over_four_axes[gpu_backend](src, dst)
    np.testing.assert_array_equal(dst[:, 0, 0, 0], src[:, 0, 0, 0])


@kernel(domain="src")
def work_size_over_three_axes(src: np.ndarray, dst: np.ndarray):
    """The largest domain a grid can carry, reading every axis."""
    i = get_global_id(0)
    j = get_global_id(1)
    k = get_global_id(2)
    dst[i, j, k] = src[i, j, k]


def test_work_size_over_three_axes_still_runs(gpu_backend):
    """A rank-3 declaration is the boundary case and must keep working."""
    shape = (2, 3, 4)
    src = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    dst = np.zeros(shape, dtype=np.float32)
    work_size_over_three_axes[gpu_backend](src, dst)
    np.testing.assert_array_equal(dst, src)


