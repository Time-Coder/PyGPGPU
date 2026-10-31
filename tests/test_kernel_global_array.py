"""The audit, as tests: a module-global array written by a magic loop."""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


GLOBAL_OUT = np.zeros(4, dtype=np.int32)


@kernel
def write_global(x: np.ndarray):
    """Element writes to a module global the launch never received as an argument."""
    for i in range(x.shape[0]):
        GLOBAL_OUT[i] = int(x[i]) * 10


def test_element_writes_reach_the_global():
    """The global travels as a parameter, and the writes land in the caller's object."""
    GLOBAL_OUT[:] = 0
    values = np.arange(1, 5, dtype=np.int32)
    write_global["cuda"](values)
    assert GLOBAL_OUT.tolist() == [10, 20, 30, 40]


GLOBAL_ACC = np.zeros(4, dtype=np.int32)


@kernel
def add_into_global(x: np.ndarray):
    """Augmented element writes accumulate across launches."""
    for i in range(x.shape[0]):
        GLOBAL_ACC[i] += int(x[i])


def test_augmented_writes_accumulate_across_launches():
    """The read-modify-write reads what the previous launch left."""
    GLOBAL_ACC[:] = 0
    values = np.arange(1, 5, dtype=np.int32)
    add_into_global["cuda"](values)
    assert GLOBAL_ACC.tolist() == [1, 2, 3, 4]
    add_into_global["cuda"](values)
    assert GLOBAL_ACC.tolist() == [2, 4, 6, 8]


GLOBAL_SUM = np.zeros(1, dtype=np.int32)


@kernel
def sum_into_global(x: np.ndarray):
    """Every work item adds into one shared slot: the reduction shape."""
    for i in range(x.shape[0]):
        GLOBAL_SUM[0] += int(x[i])


def test_a_reduction_into_a_global_slot_is_exact():
    """One location, updated atomically, read back once the launch finishes."""
    GLOBAL_SUM[:] = 0
    values = np.arange(1, 5, dtype=np.int32)
    sum_into_global["cuda"](values)
    assert int(GLOBAL_SUM[0]) == 10


GLOBAL_TAIL = np.zeros(4, dtype=np.int32)


@kernel
def tail_write(x: np.ndarray):
    """The tail runs on the host, so its write lands in the caller's object directly."""
    for i in range(x.shape[0]):
        GLOBAL_TAIL[i] = int(x[i])
    GLOBAL_TAIL[0] = 999


def test_a_tail_write_into_the_global_is_seen():
    """The tail runs on the host, so its write lands in the caller's object directly."""
    GLOBAL_TAIL[:] = 0
    values = np.arange(1, 5, dtype=np.int32)
    tail_write["cuda"](values)
    assert GLOBAL_TAIL.tolist() == [999, 2, 3, 4]


_REBIND_TARGET = np.zeros(4, dtype=np.int32)


def test_rebinding_the_global_is_refused():
    """The array object itself is shared with the scope: rebinding it is refused."""

    @kernel
    def rebind(x: np.ndarray):
        global _REBIND_TARGET
        for i in range(x.shape[0]):
            _REBIND_TARGET = x

    values = np.arange(1, 5, dtype=np.int32)
    with pytest.raises(SyntaxError) as caught:
        rebind["cuda"](values)
    assert "rebind the outer array" in str(caught.value)
