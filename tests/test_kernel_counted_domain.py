"""A runtime count is an iteration domain: `for i in range(n)` is one work item per sample.

The rule is *who decides the count*, not what the loop looks like:

* a **shape** -- `range(arr.shape[0])`, `enumerate(arr)`, `np.ndindex(arr.shape)` -- is a
  property of the caller's data, and has always been the dispatch;
* a **runtime count** -- `range(n)` with `n` a kernel parameter -- is a value only the call
  knows. It is the same kind of domain, and promoting it is what stops a body whose only bound
  is a scalar from running on a single thread;
* a **compile-time count** -- `range(8)`, `range(MAX_ITER)` -- is a fixed number of steps one
  work item walks, and stays a serial loop.

The motivating body is a Monte Carlo estimator: `for i in range(n)` with `n` samples. It used
to compile to one work item looping a hundred thousand times, which is correct output on one
thread -- nothing in the source looked wrong, and the launch log said `(1, 1, 1)`.

The second half of this module pins the parts that have to *agree* for that to be safe: the
host-region split (which statement is above the dispatch), the several-pass splitter (a second
counted loop is a second pass), and the integer requirement (python's `range()` refuses a
float). The failure of any one of them is silent: a statement lands in the wrong region, or a
launch races on one output slot.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# The dispatch itself
# ---------------------------------------------------------------------------


@kernel
def counted_fill(out: np.ndarray, n: int):
    """One work item per count: each writes the slot its own index owns."""
    for i in range(n):
        out[i] = float(i) * 2.0


@kernel
def literal_walk(out: np.ndarray, n: int):
    """A literal count is one work item walking eight steps, not eight work items.

    The walk sits *inside* the dispatch so its result is observable: a body-level serial loop
    above the dispatch is host work and is not translated (a pre-existing rule this change
    does not touch -- see `probe_control_shape`, where the shape spelling behaves identically).
    """
    for j in range(n):
        total = 0
        for i in range(8):
            total += 1
        out[j] = float(total)


def test_a_counted_domain_launches_one_work_item_per_count(runner):
    """The whole point: `n` items are computed by `n` work items."""
    for n in (1, 7, 64, 1000):
        out = np.zeros(n, dtype=np.float32)
        runner(counted_fill, out, np.int32(n))
        np.testing.assert_allclose(out, np.arange(n, dtype=np.float32) * 2.0)


def test_a_counted_domain_reads_its_count_per_call(runner):
    """The count is a call-time value, not a size frozen at compile time."""
    for n in (5, 33, 5):
        out = np.zeros(n, dtype=np.float32)
        runner(counted_fill, out, np.int32(n))
        np.testing.assert_allclose(out, np.arange(n, dtype=np.float32) * 2.0)


def test_a_literal_count_stays_inside_one_work_item(runner):
    """`range(8)` is a walk one work item runs, so it is not promoted to eight items."""
    out = np.zeros(4, dtype=np.float32)
    runner(literal_walk, out, np.int32(4))
    np.testing.assert_allclose(out, np.full(4, 8.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# A reduction still needs its shared slot
# ---------------------------------------------------------------------------


@kernel
def counted_reduction(x: np.ndarray, n: int) -> float:
    """`total` is initialized above the dispatch and accumulated inside it."""
    total = 0.0
    for i in range(n):
        total += float(x[i])
    return total


def test_a_counted_domain_still_carries_its_accumulator(runner):
    """The accumulator above the loop stays one shared location across the whole launch.

    This is the half that makes a promoted domain usable rather than merely parallel: with
    `n` work items the accumulator must be an atomic slot, or the answer is one item's
    partial sum. It was already carried for a shape-sourced loop; the counted loop reaches
    the same `_carried_scalars` scan only because every region split agrees on where the
    dispatch is.
    """
    n = 1000
    x = np.ones(n, dtype=np.float32)
    assert runner(counted_reduction, x, np.int32(n)) == pytest.approx(float(n))


# ---------------------------------------------------------------------------
# Python's own refusal: `range()` takes an integer
# ---------------------------------------------------------------------------


@kernel
def float_count(out: np.ndarray, n: float):
    for i in range(n):
        out[0] = 1.0


def test_a_float_count_is_refused_the_way_python_refuses_it(gpu_backend):
    """`range(4.0)` is python's `TypeError`, so the device says the same thing.

    Accepting it was worse than refusing: the count reached the guard as a float comparison,
    so every work item ran the whole body and they raced on one output slot -- a wrong answer
    from a launch that reported success.
    """
    with pytest.raises((SyntaxError, TypeError)) as caught:
        float_count[gpu_backend](np.zeros(1, dtype=np.float32), 4.0)
    message: str = str(caught.value)
    assert "integer" in message, message
    assert "range(n)" in message, message


# ---------------------------------------------------------------------------
# A second counted loop is a second pass, not a race
# ---------------------------------------------------------------------------


@kernel
def two_counted_passes(x: np.ndarray, out: np.ndarray, n: int):
    """Two counted domains: the second must run *after* the first, once each."""
    for i in range(n):
        out[i] = x[i] + 1.0
    for j in range(n):
        out[j] = out[j] * 2.0


def test_a_second_counted_domain_is_a_second_pass(runner):
    """Each pass is its own launch, so the second reads what the first wrote.

    Promoting a second counted loop without also teaching the several-pass splitter about it
    would run both loops inside every work item -- each item doing its own element twice, in
    the wrong order -- and the answer would only be right by coincidence.
    """
    n = 256
    x = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    runner(two_counted_passes, x, out, np.int32(n))
    np.testing.assert_allclose(out, (np.arange(n, dtype=np.float32) + 1.0) * 2.0)


# ---------------------------------------------------------------------------
# Statements above a counted dispatch are still host work
# ---------------------------------------------------------------------------


@kernel
def creation_above_counted(out: np.ndarray, n: int):
    """A creation above the counted loop is built once on the host, not per work item."""
    scratch: np.ndarray = np.full(n, 3.0, dtype=np.float32)
    for i in range(n):
        out[i] = scratch[i] + float(i)


def test_a_creation_above_a_counted_domain_is_host_work(runner):
    """The host region ends at the dispatch, and the dispatch is now the counted loop.

    This is the silent failure the region split is checked against: if the host pass kept
    reading the body as "no dispatch here", the creation would be translated as device code
    (there is no numpy there) instead of travelling as a parameter.
    """
    n = 128
    out = np.zeros(n, dtype=np.float32)
    runner(creation_above_counted, out, np.int32(n))
    np.testing.assert_allclose(out, np.full(n, 3.0, dtype=np.float32) + np.arange(n, dtype=np.float32))


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------


def test_the_predicate_promotes_a_parameter_and_not_a_literal():
    """`params` is what separates the two, which is why it is a required argument."""
    import ast

    from pygpgpu.kernel.translator.dispatch_domain import _is_magic_loop_stmt

    counted = ast.parse("for i in range(n):\n    pass\n").body[0]
    literal = ast.parse("for i in range(8):\n    pass\n").body[0]
    stepped = ast.parse("for i in range(0, n, 2):\n    pass\n").body[0]
    shape = ast.parse("for i in range(x.shape[0]):\n    pass\n").body[0]
    params = {"n", "x"}
    assert _is_magic_loop_stmt(counted, params)
    assert _is_magic_loop_stmt(shape, params)
    assert not _is_magic_loop_stmt(literal, params)
    assert not _is_magic_loop_stmt(stepped, params)
    # Without the parameter set a counted loop is indistinguishable from a walk, which is
    # exactly why no caller is allowed to omit it.
    assert not _is_magic_loop_stmt(counted, frozenset())


def test_a_counted_domain_reaches_the_generated_source():
    """The dispatch is emitted as a gid rather than a `for`, and the count is the guard."""
    from pygpgpu.kernel.translator import FunctionTranslator, infer_ctype

    raw = counted_fill.raw_func
    argument_types = [infer_ctype(np.zeros(4, np.float32)), infer_ctype(np.int32(4))]
    translator = FunctionTranslator(raw, argument_types, "opencl",
                                    lambda name: raw.__globals__.get(name), {},
                                    is_kernel=True)
    translator.translate()
    body: str = "\n".join(source for _, source in translator.render_kernels("counted_fill"))
    assert "get_global_id(0)" in body
    assert "for (int" not in body, body
    # The parameter's own value is the bound, carried as a launch parameter.
    assert translator.work_size_dims == [("n", -2)] or translator.work_size_dims == [("n", 0)]
