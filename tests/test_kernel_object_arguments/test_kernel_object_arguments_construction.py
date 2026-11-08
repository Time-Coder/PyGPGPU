"""The construction tests, split out of `test_kernel_object_arguments.py`.

The subject is construction.
"""
from __future__ import annotations

from _helpers.test_kernel_object_arguments_helpers import (
    Greeter,
    Pair,
    constructs_inline,
    constructs_named,
    reads_two_members,
    through_a_helper,
)

import numpy as np
import pytest


def test_a_construction_written_into_a_call(runner):
    """`sum_of(Pair(1.0, 2.0))` builds the value inline, and the scan has to see it.

    An unnamed temporary has no path for the scan to bind, so the value was refused with
    "device code has no objects"; the tree is now read as though the body had written the
    name itself. The value also travels by address, and a compound literal has no address,
    so a temporary local is emitted for the argument.
    """
    out = np.zeros(1, dtype=np.float32)
    runner(constructs_inline, out)
    assert float(out[0]) == pytest.approx(3.0)
    runner(constructs_named, out)
    assert float(out[0]) == pytest.approx(3.0)


def test_a_helper_receiving_the_object(runner):
    out = np.zeros(1, dtype=np.float32)
    runner(through_a_helper, Greeter(2.5), out)
    assert float(out[0]) == pytest.approx(7.5)


def test_unused_attributes_are_invisible():
    """Only `a` and `b` are read, so the synthesized structure has two fields -- the fact that
    `unused` exists, holding a string no device type could hold, must not matter."""
    from pygpgpu.kernel.objects import scan_object

    scan = scan_object(reads_two_members.raw_func, {"item": Pair(1.0, 2.0)},
                       lambda name: None)
    assert scan.members["item"] == ["a", "b"]
