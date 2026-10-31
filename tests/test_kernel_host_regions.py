"""The statements a body runs outside its magic loops, and the data they see.

Three behaviours together, because they share one model: python runs the statements above
the loops, between the loops, and after the last loop *once*, on the host, and only the
loop bodies become work items. Whatever a host statement reads has to be current -- an
array the previous pass wrote is pulled back before the statement runs and pushed again
for the next launch.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def tail_after_loop(x: np.ndarray, out: np.ndarray):
    """The statements after the last loop run on the host, once."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    for i in range(out.shape[0]):
        out[i] = out[i] + 100


def test_tail_statements_run_after_the_loop():
    """[1,2,3,4] doubled then shifted: the tail sees the loop's final values."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        tail_after_loop[backend](values, result)
        np.testing.assert_array_equal(result, values * 2 + 100)


@kernel
def tail_reads_the_array(x: np.ndarray, out: np.ndarray):
    """A tail statement that indexes the array the loop wrote."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    for i in range(out.shape[0]):
        out[i] = out[i] + int(out[i])


def test_tail_reads_the_arrays_the_loop_wrote():
    """out[i] = 2x + 2x: the array the loop filled is current on the host."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        tail_reads_the_array[backend](values, result)
        np.testing.assert_array_equal(result, values * 4)


@kernel
def gap_reads_the_previous_pass(x: np.ndarray, out: np.ndarray):
    """A gap statement indexes the array the previous pass wrote."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    first = int(out[0])
    for i in range(out.shape[0]):
        out[i] = out[i] + first


def test_a_gap_statement_sees_the_previous_pass():
    """out[i] += out[0]: the pull-back happened before the statement ran."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        gap_reads_the_previous_pass[backend](values, result)
        np.testing.assert_array_equal(result, values * 2 + 2)


@kernel
def gap_writes_the_next_pass_input(x: np.ndarray, out: np.ndarray):
    """A gap statement writes the array the next pass reads."""
    for i in range(x.shape[0]):
        out[i] = int(x[i])
    for i in range(out.shape[0]):
        out[i] = out[i] + 1
    for i in range(out.shape[0]):
        out[i] = out[i] * 10


def test_a_host_write_between_passes_reaches_the_next_launch():
    """The host write is not lost: the push-back happens before the next launch.

    This is the shape a stale buffer pool used to answer wrongly -- the device kept its
    own copy and the host's `+ 1` never landed.
    """
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        gap_writes_the_next_pass_input[backend](values, result)
        np.testing.assert_array_equal(result, (values + 1) * 10)


@kernel
def in_place_between_passes(x: np.ndarray, out: np.ndarray):
    """The same array as input and output across passes."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2
    for i in range(out.shape[0]):
        out[i] = out[i] + 1


def test_an_in_place_pass_chain_keeps_the_values():
    """b = 2x, then b += 1 -- the second pass reads the first pass's result."""
    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        in_place_between_passes[backend](values, result)
        np.testing.assert_array_equal(result, values * 2 + 1)
