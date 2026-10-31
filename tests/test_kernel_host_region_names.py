"""Host regions resolve names the way a function body does.

A `@kernel` returned from a factory closes over its factory's locals, and a body may
declare `nonlocal`/`global`. The host regions (above the loops, between passes, after the
last one) execute as python, so a closure cell must be visible there and a write through a
nonlocal slot must travel back to the scope.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


def make_tail_reader():
    factor = 3                       # closure cell

    @kernel
    def read_closure_in_tail(x: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.zeros(x.shape[0], dtype=np.int32)
        for i in range(x.shape[0]):
            result[i] = int(x[i])
        scaled = result[0] * factor
        result[0] = scaled
        return result

    return read_closure_in_tail


def test_a_closure_cell_is_visible_in_the_tail():
    """The tail reads `factor` the way the function body would."""
    values = np.arange(1, 5, dtype=np.int32)
    got = make_tail_reader()["cuda"](values)
    # The loop wrote [1, 2, 3, 4]; the tail set result[0] = result[0] * 3.
    assert got.tolist() == [3, 2, 3, 4]


def make_gap_writer():
    count = 0

    @kernel
    def write_in_gap(x: np.ndarray, out: np.ndarray):
        nonlocal count
        for i in range(x.shape[0]):
            out[i] = int(x[i])
        count += 1

    return write_in_gap, lambda: count


def test_a_nonlocal_write_in_a_host_region_travels_back():
    """`count += 1` after the loop updates the enclosing scope, once."""
    writer, get_count = make_gap_writer()
    values = np.arange(1, 5, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    writer["cuda"](values, out)
    assert out.tolist() == [1, 2, 3, 4]
    assert get_count() == 1
    writer["cuda"](values, out)
    assert get_count() == 2


def make_above_reader():
    factor = 3

    @kernel
    def read_above(x: np.ndarray, out: np.ndarray):
        scale = factor
        for i in range(x.shape[0]):
            out[i] = int(x[i]) * scale

    return read_above


def test_a_closure_cell_is_visible_above_the_loop():
    """The oldest path: a creation's dependency read from the enclosing scope."""
    values = np.arange(1, 5, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    make_above_reader()["cuda"](values, out)
    assert out.tolist() == [3, 6, 9, 12]
