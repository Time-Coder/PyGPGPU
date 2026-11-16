"""The returns a caller writes after the plain `return value` forms work.

Three shapes share the return machinery and each has its own way to go wrong:

* a **tuple** return -- several values of the same type used to share one slot, so the
  second store overwrote the first and the caller received `(6.0, 6.0)` where python
  answered `(3.0, 6.0)`;
* a **conditional** return -- only some work items reach it, and the one that does decides;
* a **bare** `return` -- answers None, which must not be confused with "no work item
  returned" when a later branch returns a value.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Several positions in one return site
# ---------------------------------------------------------------------------


@kernel
def pair_same_type(a: np.ndarray):
    """Two floats of the same element type: each needs its own slot."""
    for i in range(a.shape[0]):
        return a[i], a[i] * 2.0
    return 0.0, 0.0


@kernel
def pair_distinct_types(a: np.ndarray):
    """A float and an int: the two slots never shared by type, so this always worked."""
    for i in range(a.shape[0]):
        return a[i], i
    return 0.0, 0


@kernel
def triple_same_type(a: np.ndarray):
    """Three positions of one type -- the sharing bug hides behind any pair."""
    for i in range(a.shape[0]):
        return a[i], a[i] + 1.0, a[i] + 2.0
    return 0.0, 0.0, 0.0


def test_two_values_of_one_type_come_back_separately(gpu_backend):
    """`return a[i], a[i] * 2.0` must answer both values, not the second twice.

    The slot layout was keyed by the C element type, so the two float positions of one site
    shared a buffer: the second store overwrote the first and the caller received
    `(6.0, 6.0)` for an input of `3.0`. Only the launch's claim word distinguishes
    positions, so a shared slot could never answer both.
    """
    a = np.array([3.0, 4.0], dtype=np.float32)
    first, second = pair_same_type[gpu_backend][(1, 2)](a)
    assert float(first) == pytest.approx(3.0), "the first position's value was overwritten"
    assert float(second) == pytest.approx(6.0)


def test_three_values_of_one_type_come_back_separately(gpu_backend):
    """The same rule at three positions, so a pair-only fix would not pass this."""
    a = np.array([5.0, 6.0], dtype=np.float32)
    result = triple_same_type[gpu_backend][(1, 2)](a)
    assert [float(v) for v in result] == pytest.approx([5.0, 6.0, 7.0])


def test_values_of_different_types_still_come_back(gpu_backend):
    """The control: distinct element types never shared a slot, and must keep working."""
    a = np.array([2.5, 3.5], dtype=np.float32)
    first, second = pair_distinct_types[gpu_backend][(1, 2)](a)
    assert float(first) == pytest.approx(2.5)
    assert int(second) == 0
