"""A host write between two launches reaches the device.

A write-only parameter used to skip the upload entirely: the device buffer came from the
pool with whatever the previous launch left in it, so the host's version of the data was
silently replaced by unrelated numbers.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


@kernel
def fill(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        out[i] = int(x[i]) * 2


@kernel
def add_from(x: np.ndarray, out: np.ndarray):
    """out = x + out: reads the device copy of `out` that the caller wrote on the host."""
    for i in range(x.shape[0]):
        out[i] = int(x[i]) + int(out[i])


def test_a_host_write_between_launches_reaches_the_device():
    """The second launch reads what the host wrote, not what the pool had.

    First launch: result = 2 * values = [20, 22, 24, 26]. The host then writes +1, and the
    second launch adds `values` into what is on the device -- which must be the host's
    version, [21, 23, 25, 27], not the pool's stale contents.
    """
    values = np.arange(10, 14, dtype=np.int32)
    result = np.zeros(4, dtype=np.int32)
    fill["cuda"](values, result)
    assert result.tolist() == [20, 22, 24, 26]
    result += 1                      # host writes [21, 23, 25, 27]
    add_from["cuda"](values, result)
    assert result.tolist() == [31, 34, 37, 40]   # values + (2*values + 1)


def test_two_launches_chain_without_host_writes():
    """Even with no host write, the second launch must see the first one's result."""
    values = np.arange(10, 14, dtype=np.int32)
    result = np.zeros(4, dtype=np.int32)
    fill["cuda"](values, result)
    add_from["cuda"](values, result)
    assert result.tolist() == [30, 33, 36, 39]   # values + 2*values, elementwise
