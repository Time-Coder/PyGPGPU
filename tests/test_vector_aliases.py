"""The semantic aliases are exported from pygpgpu: one import domain for users."""
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu
import pygf
from pygpgpu import color3f, kernel, point3f


ALL_ALIASES = [
    "color3d", "color3f", "color4d", "color4f",
    "normal3d", "normal3f", "point3d", "point3f",
    "texCoord2d", "texCoord2f", "texCoord3d", "texCoord3f",
    "vector3d", "vector3f",
    "vec2", "vec3", "vec4",
]


@pytest.mark.parametrize("name", ALL_ALIASES)
def test_an_alias_is_exported_and_is_pygf_s_object(name: str) -> None:
    """One object per name: pygpgpu re-exports pygf's class, never a copy."""
    exported = getattr(pygpgpu, name)
    assert exported is getattr(pygf, name)
    assert name in pygpgpu.__all__


def test_the_fp16_and_matrix_aliases_stay_pygf_only() -> None:
    """The device backends do not carry fp16 vectors, and matrices are their own shape."""
    for name in ("color3h", "point3h", "texCoord2h", "vector3h"):
        assert not hasattr(pygpgpu, name), name
    for name in ("matrix4f", "quatf", "frame4d"):
        assert not hasattr(pygpgpu, name), name


@kernel
def alias_kernel(out: np.ndarray, c: color3f, p: point3f):
    mixed = color3f(p.x, p.y, c.r)
    out[0] = mixed.g
    out[1] = c.b


def test_a_kernel_builds_and_takes_aliases_without_importing_pygf() -> None:
    """`color3f`/`point3f` work as parameter types and constructors, on the device."""
    out = np.zeros(2)
    alias_kernel["cuda"](out, color3f(0.25, 0.5, 0.75), point3f(1.0, 2.0, 3.0))
    np.testing.assert_allclose(out, [2.0, 0.75])
