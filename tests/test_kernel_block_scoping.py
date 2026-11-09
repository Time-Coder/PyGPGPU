"""Python's block statements share one function scope, and the device must agree.

Only `def`/`class`/lambda/comprehension create a scope in python; `if`, `elif`, `while` and
`for` do not -- a name first assigned inside any of them is a function local, readable after
the block. The translator used to record a binding on the *block* being translated, so it
vanished when the block ended; the binding now lands on the function's own scope and the C
declaration is hoisted to the top of the function, which is block-agnostic by construction:
every braced construct goes through `_push_scope_block`, and the hoist tests the nesting
depth, not the statement kind.

These tests walk the constructs one by one, because "the mechanism covers them" is exactly
the kind of claim a single `if` test cannot carry.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


@kernel
def from_a_while_block(out: np.ndarray, limit: int):
    for i in range(out.shape[0]):
        value = 0.0
        steps = 0
        while steps < limit:
            value = float(steps)
            steps += 1
        out[i] = value


@kernel
def from_an_elif_chain(out: np.ndarray, which: int):
    for i in range(out.shape[0]):
        if which == 0:
            picked = 1.0
        elif which == 1:
            picked = 2.0
        else:
            picked = 3.0
        out[i] = picked


@kernel
def from_a_loop_target(out: np.ndarray, count: int):
    for i in range(out.shape[0]):
        for k in range(count):
            pass
        out[i] = float(k)          # the loop's own target, read after the loop


@kernel
def from_three_levels_of_nesting(out: np.ndarray, which: int):
    for i in range(out.shape[0]):
        while which > 0:
            if which == 1:
                found = 11.0
            else:
                for k in range(2):
                    found = 22.0
            break
        out[i] = found


def test_a_while_block_introduces_the_name(gpu_backend: str) -> None:
    out = np.zeros(4, dtype=np.float32)
    from_a_while_block[gpu_backend](out, 3)
    assert np.allclose(np.asarray(out), np.float32(2.0))


def test_an_elif_chain_introduces_the_name(gpu_backend: str) -> None:
    for which, expected in ((0, 1.0), (1, 2.0), (2, 3.0)):
        out = np.zeros(4, dtype=np.float32)
        from_an_elif_chain[gpu_backend](out, which)
        assert np.allclose(np.asarray(out), np.float32(expected)), which


def test_a_loop_target_survives_the_loop(gpu_backend: str) -> None:
    out = np.zeros(4, dtype=np.float32)
    from_a_loop_target[gpu_backend](out, 3)
    assert np.allclose(np.asarray(out), np.float32(2.0))


def test_deep_nesting_reaches_the_function_scope(gpu_backend: str) -> None:
    for which, expected in ((1, 11.0), (2, 22.0)):
        out = np.zeros(4, dtype=np.float32)
        from_three_levels_of_nesting[gpu_backend](out, which)
        assert np.allclose(np.asarray(out), np.float32(expected)), which


def test_the_interpreter_reads_the_same_names() -> None:
    """The `python` backend runs the body for real: it is the reference for every case."""
    out = np.zeros(4, dtype=np.float32)
    from_a_while_block["python"](out, 3)
    assert np.allclose(out, np.float32(2.0))

    out = np.zeros(4, dtype=np.float32)
    from_a_loop_target["python"](out, 3)
    assert np.allclose(out, np.float32(2.0))


def test_statements_python_has_no_device_form_are_refused_wholesale() -> None:
    """`try`/`with`/comprehensions are a feature gap, not a scoping question: the walk
    refuses the statement itself, so no binding question can arise."""

    @kernel
    def uses_try(out: np.ndarray):
        for i in range(out.shape[0]):
            try:
                out[i] = 1.0
            except Exception:
                out[i] = 0.0

    with pytest.raises(SyntaxError) as caught:
        uses_try["cuda"][(1, 4)](np.zeros(4, dtype=np.float32))
    assert "unsupported statement" in str(caught.value), str(caught.value)

