"""A local first assigned inside a branch, and what the reader is told about it.

Python scopes a name to the whole function: assigning it inside an `if` makes it local, and
reading it before that assignment raises `UnboundLocalError` -- but only when the branch did
not run. The device has the stricter rule (C wants a declaration before the read, and the
generated declaration sits where the first assignment is), so the same source is refused.

The refusal is the subject here, not the rule: the message has to say that the *assignment is
inside the branch*, because "unknown name" sends a reader looking for a typo in a name the
body obviously assigns. When the declaration is allowed to move to the top of the function --
which is what python's function-scope rule already means -- this test flips to asserting the
value instead, and the name of the test says so.
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


def test_the_message_names_the_branch_rather_than_the_name(gpu_backend: str) -> None:
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        assigned_in_both_branches[gpu_backend](out, 1)
    message = str(caught.value)
    assert "assigned inside a branch" in message, message
    assert "value = 0" in message, message
    # The old wording pointed at the name, which is the one thing that was not wrong.
    assert "unknown name" not in message, message


def test_one_branch_is_the_same_story(gpu_backend: str) -> None:
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        assigned_in_one_branch[gpu_backend](out, 1)
    assert "assigned inside a branch" in str(caught.value)


def test_assigning_before_the_branch_is_accepted(gpu_backend: str) -> None:
    """The workaround the message points at has to work, or the message is a dead end."""
    out = np.zeros(4, dtype=np.float32)
    assigned_before_the_branch[gpu_backend](out, 1)
    assert np.allclose(np.asarray(out), np.float32(1.5))
