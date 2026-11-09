"""A local first assigned inside a branch: python's function scope, on the device.

Python scopes a name to the whole function -- assigning it inside an `if` makes it local and
readable after the `if` -- and the device used to disagree: the binding was recorded on the
block, so it vanished when the block ended and the read failed with "unknown name". Two
layers had to follow: the binding is recorded on the function's own scope, and the C
declaration moves to the top of the function (a declaration inside the block is not in scope
after it).

One place still differs from python, and it is pinned below rather than hidden: python raises
`UnboundLocalError` when the branch did not run, while a device has no exception to raise, so
the hoisted declaration starts at zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def assigned_in_both_branches(out: np.ndarray, flag: int):
    for i in range(out.shape[0]):
        if flag > 0:
            value = 1.5
        else:
            value = -1.5
        out[i] = value


@kernel
def assigned_in_one_branch(out: np.ndarray, flag: int):
    for i in range(out.shape[0]):
        if flag > 0:
            value = 1.5
        out[i] = value


@kernel
def assigned_before_the_branch(out: np.ndarray, flag: int):
    for i in range(out.shape[0]):
        value = 0.0
        if flag > 0:
            value = 1.5
        out[i] = value


@kernel
def assigned_inside_a_loop_then_read(out: np.ndarray):
    for i in range(out.shape[0]):
        total = 0.0
        for k in range(3):
            last = 2.0 * k
            total += last
        out[i] = total + last


def test_each_branch_supplies_the_value(gpu_backend: str) -> None:
    positive = np.zeros(4, dtype=np.float32)
    assigned_in_both_branches[gpu_backend](positive, 1)
    assert np.allclose(np.asarray(positive), np.float32(1.5))

    negative = np.zeros(4, dtype=np.float32)
    assigned_in_both_branches[gpu_backend](negative, -1)
    assert np.allclose(np.asarray(negative), np.float32(-1.5))


def test_assigning_before_the_branch_still_wins(gpu_backend: str) -> None:
    out = np.zeros(4, dtype=np.float32)
    assigned_before_the_branch[gpu_backend](out, 1)
    assert np.allclose(np.asarray(out), np.float32(1.5))


def test_a_name_from_a_loop_body_survives_the_loop(gpu_backend: str) -> None:
    """`last` is bound inside the loop and read after it, which python allows."""
    out = np.zeros(2, dtype=np.float32)
    assigned_inside_a_loop_then_read[gpu_backend](out)
    # 0 + 2 + 4 = 6 from the loop, plus the last `last` (4) read after it.
    assert np.allclose(np.asarray(out), np.float32(10.0))


def test_a_branch_that_did_not_run_leaves_zero(gpu_backend: str) -> None:
    """The documented divergence: python raises, a device reads the hoisted zero."""
    out = np.zeros(4, dtype=np.float32)
    assigned_in_one_branch[gpu_backend](out, -1)
    assert np.allclose(np.asarray(out), np.float32(0.0))


def test_the_interpreter_agrees_where_it_can() -> None:
    """The `python` backend runs the body for real, so the both-branches case must match."""
    out = np.zeros(4, dtype=np.float32)
    assigned_in_both_branches["python"](out, 1)
    assert np.allclose(out, np.float32(1.5))

    with pytest.raises(UnboundLocalError):
        assigned_in_one_branch["python"](np.zeros(4, dtype=np.float32), -1)
