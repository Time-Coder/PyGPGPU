"""The host tests, split out of `test_kernel_complex.py`.

The subject is host.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_the_host_functions_take_a_complex_scalar():
    """One implementation per name: the host body is pygf's, extended for complex.

    A kernel body that calls `sqrt` translates as the builtin because the object it resolves
    to *is* the builtin, and the interpreter backend runs that same body -- so the object has
    to accept a complex scalar for the two sides to agree.
    """
    import pygf

    for name, function in (("sqrt", np.sqrt), ("exp", np.exp), ("log", np.log),
                           ("sin", np.sin), ("cos", np.cos), ("tan", np.tan),
                           ("asin", np.arcsin), ("acos", np.arccos), ("atan", np.arctan),
                           ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh)):
        value = 1.0 + 2.0j
        assert getattr(pygf, name)(value) == pytest.approx(function(value))


def test_the_host_arithmetic_helpers_are_the_complex_ones():
    import pygf

    assert pygf.pow(1.0 + 2.0j, 2.0) == pytest.approx((1.0 + 2.0j) ** 2.0)
    assert pygf.conjugate(1.0 + 2.0j) == pytest.approx(1.0 - 2.0j)
    # A real pair keeps the real answer, so nothing regressed for the ordinary case.
    assert pygf.hypot(3.0, 4.0) == pytest.approx(5.0)
    with pytest.raises(TypeError):
        pygf.hypot(1.0 + 2.0j, 2.0)


def test_the_quaternions_conjugate_is_unchanged():
    import pygf
    from pygf import quatf

    flipped = pygf.conjugate(quatf(1.0, 2.0, 3.0, 4.0))
    assert (flipped.w, flipped.x, flipped.y, flipped.z) == (1.0, -2.0, -3.0, -4.0)
