"""`break` and `continue` in a kernel body.

A magic loop is not a loop in the generated source: it becomes the work-item dispatch, one
work item per iteration, with the index read from `blockIdx`/`threadIdx`. That makes the two
statements mean different things depending on **where** they sit, and the three cases are:

* **inside a real loop** -- a nested serial `for`/`while`, or `for x in arr` -- the source
  has a C loop there, so the statement is emitted as written and keeps python's meaning;
* **directly in a work item**, `continue` becomes a plain `return;`: this work item stops
  and the next one, another thread, does its own work. It must not claim the kernel result,
  which is why it is not routed through the return machinery -- a work item merely skipping
  its turn would otherwise race the value-returning sites and could answer None;
* **directly in a work item**, `break` has no meaning at all: stopping the magic loop means
  stopping every other work item, and they are already running in parallel. It is refused.

A python function run as written agrees with the first case and, by construction, with the
second one too -- `continue` only skips *this* iteration. It does **not** agree with `break`,
which is a python-level statement the device cannot express; that one is a translation error
rather than a silent divergence.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@kernel
def continue_in_a_serial_loop(x: np.ndarray, out: np.ndarray):
    """`continue` inside a nested `for`, which is a real C loop."""
    for i in range(out.shape[0]):
        total = 0.0
        for j in range(4):
            if x[j] < 0:
                continue
            total += x[j]
        out[i] = total


@kernel
def break_in_a_serial_loop(x: np.ndarray, out: np.ndarray):
    """`break` inside a nested `for`, which is a real C loop."""
    for i in range(out.shape[0]):
        total = 0.0
        for j in range(4):
            if x[j] > 5:
                break
            total += x[j]
        out[i] = total


@kernel
def continue_in_a_while(x: np.ndarray, out: np.ndarray):
    """`continue` inside a `while`, the other real loop."""
    for i in range(out.shape[0]):
        kept = 0
        j = 0
        while j < 4:
            j += 1
            if x[j - 1] < 0:
                continue
            kept += 1
        out[i] = kept


@kernel
def break_in_an_element_loop(x: np.ndarray, out: np.ndarray):
    """`break` inside `for v in arr`, which is an indexed C loop."""
    for i in range(out.shape[0]):
        found = 0.0
        for v in x:
            if v < 0:
                break
            found += v
        out[i] = found


@kernel
def continue_in_the_work_item(x: np.ndarray, out: np.ndarray):
    """`continue` directly in the magic loop's body: the work item stops here."""
    for i in range(out.shape[0]):
        if x[i] < 0:
            continue
        out[i] = x[i]


@kernel
def continue_then_a_value_return(x: np.ndarray, out: np.ndarray):
    """The shape the claim rule exists for: a `continue` next to a value-returning site.

    A work item that skips its turn must not claim, or it would decide the result with
    None; only the item that actually returns a value may win.
    """
    for i in range(out.shape[0]):
        if x[i] < 0:
            continue
        if x[i] > 5:
            return x[i]
        out[i] = x[i]


@kernel
def break_in_the_work_item(x: np.ndarray, out: np.ndarray):
    """`break` directly in the magic loop's body: refused at translation."""
    for i in range(out.shape[0]):
        if x[i] < 0:
            break
        out[i] = x[i]


# ---------------------------------------------------------------------------
# The real-loop cases keep their C meaning
# ---------------------------------------------------------------------------


def test_continue_in_a_serial_loop(runner):
    """`continue` skips the rest of that iteration, exactly as python does."""
    values = np.array([1.0, -1.0, 2.0, 3.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(continue_in_a_serial_loop, values, out)
    # Every work item sums 1 + 2 + 3: the negative element is skipped.
    np.testing.assert_allclose(out, np.full(4, 6.0))


def test_break_in_a_serial_loop(runner):
    """`break` stops that serial loop only; the other work items still run."""
    values = np.array([1.0, 2.0, 8.0, 4.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(break_in_a_serial_loop, values, out)
    # 1 + 2, then x[2] = 8 stops the loop before adding anything else.
    np.testing.assert_allclose(out, np.full(4, 3.0))


def test_continue_in_a_while(runner):
    values = np.array([1.0, -1.0, 2.0, -3.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(continue_in_a_while, values, out)
    np.testing.assert_allclose(out, np.full(4, 2.0))


def test_break_in_an_element_loop(runner):
    values = np.array([1.0, 2.0, -5.0, 4.0], dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    runner(break_in_an_element_loop, values, out)
    np.testing.assert_allclose(out, np.full(4, 3.0))


# ---------------------------------------------------------------------------
# The work-item cases
# ---------------------------------------------------------------------------


def test_continue_in_the_work_item_skips_it(runner):
    """A skipped work item writes nothing; the others write their own element."""
    values = np.array([1.0, -1.0, 2.0, -3.0], dtype=np.float32)
    out = np.full(4, -7.0, dtype=np.float32)
    runner(continue_in_the_work_item, values, out)
    assert out.tolist() == [1.0, -7.0, 2.0, -7.0]


def test_a_skipping_work_item_does_not_claim_the_result(runner):
    """The value site wins, not the item that only skipped its turn.

    A `continue` emitted as a claiming `return` would have let a skipping work item take the
    result with None. The negative elements are exactly the ones that skip, and the single
    positive-over-5 element is the one that returns.
    """
    values = np.array([-1.0, -2.0, 9.0, -3.0], dtype=np.float32)
    out = np.full(4, -7.0, dtype=np.float32)
    assert float(runner(continue_then_a_value_return, values, out)) == 9.0
    assert out.tolist() == [-7.0, -7.0, -7.0, -7.0]


def test_break_in_the_work_item_is_refused(gpu_backend, runner):
    """There is no device meaning for "stop the whole magic loop", so it is a SyntaxError."""
    values = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(break_in_the_work_item, values, out)
    message = str(caught.value)
    # The message says what to do instead, and names the parallel reason.
    assert "break" in message
    assert "parallel" in message
    assert "range(" in message


# ---------------------------------------------------------------------------
# The generated source
# ---------------------------------------------------------------------------


def _source(func, backend):
    """The generated device source for one kernel, with two float arrays as its types."""
    from pygpgpu.kernel.decorator import infer_ctype
    from pygpgpu.kernel.translator import FunctionTranslator

    raw = func.raw_func if hasattr(func, "raw_func") else func
    f = infer_ctype(np.zeros(4, dtype=np.float32))
    translator = FunctionTranslator(raw, [f, f], backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n".join(piece for _, piece in translator.render_kernels("k"))


def test_the_work_item_continue_becomes_a_plain_return():
    """Not a `continue;` -- the generated function body has no loop to continue."""
    source = _source(continue_in_the_work_item, "cuda")
    assert "continue;" not in source
    # The skip is the guard's `return;`, the same statement the bounds check already uses.
    assert source.count("return;") >= 2


def test_a_real_loop_keeps_its_own_continue_and_break():
    source = _source(continue_in_a_serial_loop, "cuda")
    assert "continue;" in source
    source = _source(break_in_a_serial_loop, "cuda")
    assert "break;" in source


def test_a_skipping_work_item_never_claims():
    """The `continue` must not be compiled into a compare-and-swap claim.

    The claim is what decides the result, so a skipping item that claims turns a well-defined
    answer into None. The value site is the only claimant here.
    """
    source = _source(continue_then_a_value_return, "cuda")
    # One claim for the value site, and none for the guard's `continue`.
    assert source.count("atomicCAS") == 1
