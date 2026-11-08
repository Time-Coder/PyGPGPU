"""The widths tests, split out of `test_kernel_object_arguments.py`.

The subject is widths.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Pair,
    two_objects,
    widens_across_instances,
)

import numpy as np
import pytest
from pygpgpu import kernel


def test_two_objects_of_one_class(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(two_objects, Pair(1.0, 2.0), Pair(10.0, 20.0), out)
    assert float(out[0]) == pytest.approx(21.0)


def test_int_widens_to_float(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(widens_across_instances, Pair(1, 1.0), Pair(2.5, 2.0), out)
    assert float(out[0]) == pytest.approx(3.5)


def test_two_instances_stay_two_structures_when_one_lacks_a_member():
    """`left` only needs `a`, `right` needs `b` which `left` does not carry: merging would
    invent a member one of the two does not have, so they keep separate structures."""
    from pygpgpu.kernel.objects import convert_object_arguments

    class Lopsided:
        def __init__(self, **members):
            for name, value in members.items():
                setattr(self, name, value)

    @kernel
    def differing(left, right, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = left.a + right.b

    converted = convert_object_arguments(
        differing.raw_func, (Lopsided(a=1.0), Lopsided(b=2.0)), lambda name: None)
    assert type(converted[0]).__name__ != type(converted[1]).__name__


def test_two_instances_stay_two_structures_when_types_do_not_promote():
    """`a` is a float in one instance and a bool in the other: no type holds both honestly."""
    from pygpgpu.kernel.objects import convert_object_arguments

    converted = convert_object_arguments(
        widens_across_instances.raw_func, (Pair(1.5, 2.0), Pair(True, 2.0)), lambda name: None)
    assert type(converted[0]) is not type(converted[1])
