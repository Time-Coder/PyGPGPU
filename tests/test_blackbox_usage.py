"""Black-box probes: the call shapes a user actually writes, run to see what breaks.

This is the second discovery pass, from the outside in: instead of reading the translator,
write the usage a reader of the README would write and see whether the answer is python's.
Each probe is a *question* first; only the ones that answered wrongly became tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# --- probe 1: a kernel that returns a value computed inside the loop -------------------


@kernel
def first_negative(a: np.ndarray):
    """Return the first negative element's index, as a python `for`-with-return would."""
    for i in range(a.shape[0]):
        if a[i] < 0:
            return i
    return -1


def test_a_return_inside_a_loop_answers_the_first_match(gpu_backend):
    """`return` inside the loop is python's early exit: the first work item to reach it wins."""
    a = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float32)
    result = first_negative[gpu_backend][(1, 4)](a)
    assert result == 2


# --- probe 2: an array argument modified in place is visible to the caller --------------


@kernel
def double_in_place(a: np.ndarray):
    for i in range(a.shape[0]):
        a[i] = a[i] * 2.0


def test_in_place_write_reaches_the_caller(gpu_backend):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    double_in_place[gpu_backend](a)
    np.testing.assert_array_equal(a, np.array([2.0, 4.0, 6.0], dtype=np.float32))


# --- probe 3: two arrays aliased to the same buffer -------------------------------------


@kernel
def add_into(a: np.ndarray, b: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + b[i]


def test_the_same_array_passed_twice(gpu_backend):
    """`k(x, x, out)` -- one buffer under two names -- must still add elementwise."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    add_into[gpu_backend](x, x, out)
    np.testing.assert_array_equal(out, np.array([2.0, 4.0, 6.0], dtype=np.float32))


# --- probe 4: writing an output in a *different* order than the index -------------------


@kernel
def reverse_into(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[out.shape[0] - 1 - i]


def test_a_reversed_read_is_not_a_race(gpu_backend):
    """Reading backwards and writing forwards: no two work items write the same element."""
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    reverse_into[gpu_backend][(1, 4)](a, out)
    np.testing.assert_array_equal(out, np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32))


# --- probe 5: a scalar computed on the host from an argument ---------------------------


@kernel
def scale_by_len(rows: int, out: np.ndarray):
    """A host statement above the loop: the loop bound comes from an argument, not a shape."""
    shift = rows * 2
    for i in range(out.shape[0]):
        out[i] = shift


def test_a_host_value_from_an_argument(gpu_backend):
    out = np.zeros(2, dtype=np.float32)
    scale_by_len[gpu_backend](3, out)
    np.testing.assert_array_equal(out, np.array([6.0, 6.0], dtype=np.float32))


# --- probe 6: a 2-D magic loop over indices --------------------------------------------


@kernel
def transpose(a: np.ndarray, out: np.ndarray):
    for i, j in np.ndindex(a.shape):
        out[j, i] = a[i, j]


def test_an_ndindex_transpose(gpu_backend):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros((3, 2), dtype=np.float32)
    transpose[gpu_backend](a, out)
    np.testing.assert_array_equal(out, a.T)


# --- probe 7: the bracket a user guesses for a 2-D problem ------------------------------


def test_a_2d_bracket_on_a_2d_loop(gpu_backend):
    """A per-axis bracket states one block and one thread count per axis.

    The ranks must match, which the parser says plainly when they do not; this pins the
    spelling that works and that the loop still tiles the array correctly under it.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros((3, 2), dtype=np.float32)
    transpose[gpu_backend][((1, 1), (2, 3))](a, out)
    np.testing.assert_array_equal(out, a.T)


def test_a_mismatched_rank_bracket_is_refused_in_words_that_say_why(gpu_backend):
    """The refusal names both ranks, so a caller can see which half to fix."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-dimensional block count"):
        transpose[gpu_backend][(1, (2, 3))](a, out)
