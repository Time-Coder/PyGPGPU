"""The statements after the LAST magic loop run once, on the host.

This module pins the tail specifically: a body with *one* loop and statements after it.
A two-loop body exercises the between-pass path instead, which is a different mechanism.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def tail_reduces(x: np.ndarray, out: np.ndarray):
    """A tail statement only the host can run: np.sum over the loop's result."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    total = int(np.sum(out))
    out[0] = total


def test_a_tail_statement_only_the_host_can_run():
    """np.sum has no device form: the tail ran on the host, after the launch."""
    values = np.arange(1, 5, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    tail_reduces["cuda"](values, out)
    # The loop wrote [2, 4, 6, 8]; the tail overwrote out[0] with their sum.
    assert out.tolist() == [20, 4, 6, 8]


@kernel
def tail_indexes_the_result(x: np.ndarray, out: np.ndarray):
    """A tail statement that indexes the array the loop wrote."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    out[0] = int(out[3]) + 10


def test_a_tail_statement_indexes_the_loops_result():
    """out[3] is the loop's final value by the time the tail runs."""
    values = np.arange(1, 5, dtype=np.int32)
    out = np.zeros(4, dtype=np.int32)
    tail_indexes_the_result["cuda"](values, out)
    # The loop wrote [2, 4, 6, 8]; the tail set out[0] = 8 + 10.
    assert out.tolist() == [18, 4, 6, 8]


@kernel
def tail_uses_a_carried_value(x: np.ndarray):
    """The loop accumulates into a shared slot; the tail returns it."""
    evens = 0
    for i in range(x.shape[0]):
        if int(x[i]) % 2 == 0:
            evens += 1
    count = evens
    return count


def test_a_tail_return_uses_the_carried_value():
    """The shared slot's final value is what the tail's return hands back."""
    values = np.array([1, 3, 6, 7], dtype=np.int32)
    assert int(tail_uses_a_carried_value["cuda"](values)) == 1
