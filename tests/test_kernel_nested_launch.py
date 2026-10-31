"""A kernel body that launches another kernel (CUDA's dynamic parallelism).

This is the one feature that needs a *device-code link*: the child kernel is resolved
through the device runtime, which NVRTC does not link in, so a source that nests a launch
is built by nvcc instead. Everything else keeps using NVRTC, and the tests below therefore
mark CUDA specifically rather than going through `gpu_backend`.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import get_global_id, kernel


@kernel
def child(out: np.ndarray):
    """Doubles its index into ``out``; launched from inside another kernel."""
    out[get_global_id(0)] = get_global_id(0) * 2


@kernel
def parent(out: np.ndarray):
    """A work item launches ``child`` over the same span."""
    child[(out.shape[0],)](out)


@kernel
def parent_with_group(out: np.ndarray):
    """The bracket's second entry is the local work size."""
    child[(out.shape[0], 4)](out)


@kernel
def two_children(first: np.ndarray, second: np.ndarray):
    """Two launches of one kernel: the definition is shared, not duplicated."""
    child[(first.shape[0],)](first)
    child[(second.shape[0],)](second)


def test_a_kernel_can_launch_a_kernel():
    """The child's writes come back, so the device-side launch really ran."""
    out = np.zeros(8, dtype=np.int32)
    parent["cuda"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


def test_a_nested_launch_takes_a_work_size_and_a_group():
    """`child[(n, local)]` splits the same span, exactly as a host launch would."""
    out = np.zeros(8, dtype=np.int32)
    parent_with_group["cuda"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


def test_two_launches_of_one_kernel_share_its_definition():
    """Both arrays are filled, and the child is emitted once rather than per call site."""
    first = np.zeros(4, dtype=np.int32)
    second = np.zeros(4, dtype=np.int32)
    two_children["cuda"](first, second)
    np.testing.assert_array_equal(first, np.arange(4, dtype=np.int32) * 2)
    np.testing.assert_array_equal(second, np.arange(4, dtype=np.int32) * 2)


def test_the_interpreter_runs_a_nested_launch_too():
    """A serial run has no device to launch onto, so the call is just a call."""
    out = np.zeros(8, dtype=np.int32)
    parent["python"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


@pytest.mark.parametrize("backend", ["opencl", "hip"])
def test_a_backend_without_nested_launch_says_so(backend):
    """OpenCL has no nested-launch statement, and HIP does not offer it here."""
    with pytest.raises(SyntaxError) as caught:
        parent[backend](np.zeros(8, dtype=np.int32))
    message: str = str(caught.value)
    assert "dynamic parallelism" in message


def test_a_kernel_called_without_a_work_size_is_refused():
    """From device code a @kernel is a launch, so a plain call is a mistake worth naming."""

    @kernel
    def calls_a_kernel_as_a_helper(out: np.ndarray):
        child(out)

    with pytest.raises(SyntaxError) as caught:
        calls_a_kernel_as_a_helper["cuda"](np.zeros(4, dtype=np.int32))
    message: str = str(caught.value)
    assert "launch" in message
    assert "work size" in message


def test_two_magic_loops_are_two_passes():
    """Each magic loop becomes its own kernel, launched in order over one domain.

    Before this, only the first loop became the work-item dispatch and the second was emitted
    as an ordinary serial loop *inside* every work item -- so N items each ran the whole
    phase and raced on the same array, and a shape that happened to agree still read as
    correct.
    """

    @kernel
    def two_phases(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i] * 2
        for i in range(a.shape[0]):
            b[i] = b[i] + 1

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        two_phases[backend](values, result)
        np.testing.assert_array_equal(result, np.array([1, 3, 5, 7], dtype=np.int32))


def test_three_phases_run_in_order():
    """Three loops are three passes, each seeing what the previous one wrote."""

    @kernel
    def three_phases(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i] * 2
        for i in range(a.shape[0]):
            b[i] = b[i] + 1
        for i in range(a.shape[0]):
            b[i] = b[i] * 3

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        three_phases[backend](values, result)
        np.testing.assert_array_equal(result, np.array([3, 9, 15, 21], dtype=np.int32))


def test_a_host_only_expression_between_two_loops_runs_on_the_host():
    """The gap is python code: `np.sum(x)` over a real array has no device form at all.

    Before this, a gap statement was copied into the pass that reads it and recomputed by
    every work item -- which is wrong for a value python computes once, and impossible when
    the expression cannot be translated.
    """

    @kernel
    def host_computed(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i]
        total = int(np.sum(a))
        for i in range(a.shape[0]):
            b[i] = b[i] + total

    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        host_computed[backend](values, result)
        np.testing.assert_array_equal(result, values + int(values.sum()))


def test_two_values_between_two_loops_build_on_each_other():
    """The second gap statement reads what the first computed, on the host."""

    @kernel
    def chained(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i]
        base = int(np.max(a))
        extra = base + 1
        for i in range(a.shape[0]):
            b[i] = b[i] + extra

    values = np.arange(1, 5, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        chained[backend](values, result)
        np.testing.assert_array_equal(result, values + (int(values.max()) + 1))


def test_a_scalar_assigned_between_two_loops_reaches_the_second():
    """A plain assignment between the passes is copied into the pass that reads it."""

    @kernel
    def scaled(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i] * 2
        factor = 10
        for i in range(a.shape[0]):
            b[i] = b[i] + factor

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        scaled[backend](values, result)
        np.testing.assert_array_equal(result, np.array([10, 12, 14, 16], dtype=np.int32))


def test_an_array_created_between_two_loops_is_a_parameter():
    """A creation between the passes is built once and travels as an argument."""

    @kernel
    def with_middle(a: np.ndarray, b: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i] * 2
        middle: np.ndarray = np.zeros(4, dtype=np.int32)
        for i in range(a.shape[0]):
            b[i] = b[i] + middle[i] + 1

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        with_middle[backend](values, result)
        np.testing.assert_array_equal(result, np.array([1, 3, 5, 7], dtype=np.int32))


def test_an_array_created_above_the_first_loop_is_shared():
    """A creation above the first pass is read by every pass after it."""

    @kernel
    def shared_table(a: np.ndarray, b: np.ndarray):
        table: np.ndarray = np.arange(4, dtype=np.int32)
        for i in range(a.shape[0]):
            b[i] = table[i] * 2
        for i in range(a.shape[0]):
            b[i] = b[i] + table[i]

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        result = np.zeros(4, dtype=np.int32)
        shared_table[backend](values, result)
        np.testing.assert_array_equal(result, np.array([0, 3, 6, 9], dtype=np.int32))


def test_phases_may_index_different_arrays():
    """A later pass may need a shape the first one never named."""

    @kernel
    def phases(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        for i in range(a.shape[0]):
            b[i] = a[i] * 2
        for i in range(c.shape[0]):
            c[i] = b[i] + 100

    values = np.arange(4, dtype=np.int32)
    for backend in ("python", "cuda"):
        inner = np.zeros(4, dtype=np.int32)
        outer = np.zeros(2, dtype=np.int32)
        phases[backend](values, inner, outer)
        np.testing.assert_array_equal(outer, np.array([100, 102], dtype=np.int32))


def test_an_ordinary_loop_inside_the_work_item_stays_allowed():
    """A `for` in the loop body is an ordinary loop and must keep working."""

    @kernel
    def sums(out: np.ndarray, row: np.ndarray):
        for i in range(out.shape[0]):
            total = 0
            for j in range(4):
                total = total + row[j]
            out[i] = total

    result = np.zeros(2, dtype=np.int32)
    sums["cuda"](result, np.array([1, 2, 3, 4], dtype=np.int32))
    assert result.tolist() == [10, 10]


def test_a_nested_launch_cannot_pick_a_backend():
    """The device is already running, so only the work size may be given."""

    @kernel
    def picks_a_backend(out: np.ndarray):
        child["opencl"][(out.shape[0],)](out)

    with pytest.raises(SyntaxError) as caught:
        picks_a_backend["cuda"](np.zeros(4, dtype=np.int32))
    assert "cannot be chosen" in str(caught.value)
