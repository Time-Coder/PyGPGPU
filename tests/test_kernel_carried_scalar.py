"""A scalar initialized above the magic loop and updated inside it is one location.

This is the reduction shape every tutorial starts with, and it is where a device kernel
differs from python most quietly: a plain `int total` is per work item, so N items each
accumulate one element and the "answer" is one work item's partial sum.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def sum_all(x: np.ndarray, out: np.ndarray):
    """`total` above the loop, accumulated inside it -- the whole array's sum."""
    total = 0
    for i in range(x.shape[0]):
        total += int(x[i])
    out[0] = total


@kernel
def per_item_scratch(x: np.ndarray, out: np.ndarray):
    """	otal is only used inside the loop, so it stays each work item's own."""
    total = 0
    for i in range(x.shape[0]):
        total += int(x[i])
        if i == 0:
            return total


@kernel
def only_read(x: np.ndarray, out: np.ndarray):
    """A scalar above the loop that the loop only reads stays a plain value."""
    bias = 10
    for i in range(x.shape[0]):
        out[i] = int(x[i]) + bias


def test_a_scalar_written_in_the_loop_is_one_location():
    """Every work item adds into the same slot, so the total is the whole array's."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        out = np.zeros(4, dtype=np.int32)
        sum_all[backend](values, out)
        assert int(out[0]) == int(values.sum()), (backend, out.tolist())


def test_a_scalar_used_only_inside_the_loop_stays_per_item():
    """Sharing a scratch value would turn a per-item number into a race.

    
eturn total inside the loop returns what *this* work item accumulated, and that is
    deterministic only while 	otal is private. Sharing it is reserved for a name the loop's
    result carries -- one that is read after the loop.
    """
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        out = np.zeros(4, dtype=np.int32)
        assert float(per_item_scratch[backend](values, out)) == 1.0


def test_a_read_only_scalar_above_the_loop_is_unchanged():
    """Reading it is not a write, so it needs no buffer and no atomic."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        out = np.zeros(4, dtype=np.int32)
        only_read[backend](values, out)
        np.testing.assert_array_equal(out, values + 10)
