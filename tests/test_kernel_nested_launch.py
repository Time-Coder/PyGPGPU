"""A kernel body that launches another kernel (CUDA's dynamic parallelism).

This is the one feature that needs a *device-code link*: the child kernel is resolved
through the device runtime, which NVRTC does not link in, so a source that nests a launch
is built by nvcc instead. Everything else keeps using NVRTC, and the tests below therefore
mark CUDA specifically rather than going through `gpu_backend`.
"""
from __future__ import annotations

import warnings

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


@kernel
def magic_child(a: np.ndarray, out: np.ndarray):
    """A *magic-loop* child: its extent is a derived parameter, not the body's own bound."""
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


@kernel
def magic_parent(a: np.ndarray, out: np.ndarray):
    """A magic-loop child launched from device code must forward its extent."""
    magic_child[(out.shape[0],)](a, out)


def test_a_magic_loop_child_gets_its_extent_forwarded():
    """The child's signature ends with `int _out_shape0`, and the launch has to supply it.

    A gid-style child states its work size in the bracket and derives nothing, which is why
    this case was not covered: the extent parameters are the magic loop's, and a device-side
    launch that omits them is a C compile error ("too few arguments in function call") far
    from the line that caused it.
    """
    a = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)
    magic_parent["cuda"](a, out)
    np.testing.assert_allclose(out, a * 2.0)


def test_a_magic_loop_child_forwards_the_extent_of_a_renamed_argument():
    """The argument names need not match: the child's extent follows the parent's array."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(6, dtype=np.float32)

    @kernel
    def renamed_child(source, target):
        for i in range(target.shape[0]):
            target[i] = source[i] + 1.0

    @kernel
    def renamed_parent(left, right):
        renamed_child[(right.shape[0],)](left, right)

    renamed_parent["cuda"](a, out)
    np.testing.assert_allclose(out, a + 1.0)


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
    """A child with no in-body domain has nothing for a device launch to derive a size from.

    That is the gid-style child (`out[get_global_id(0)]`): its domain is named outside the
    body -- by `@kernel(domain=...)`, which a device-side launch cannot consult -- so a bare
    call stays the "called a kernel as a helper" mistake. A *magic-loop* child does derive a
    size, and a bare call of it is a launch (see the tests below).
    """

    @kernel
    def calls_a_kernel_as_a_helper(out: np.ndarray):
        child(out)

    with pytest.raises(SyntaxError) as caught:
        calls_a_kernel_as_a_helper["cuda"](np.zeros(4, dtype=np.int32))
    message: str = str(caught.value)
    assert "launch" in message
    assert "work size" in message


@kernel
def sized_child(a: np.ndarray, out: np.ndarray):
    """A child that declares its own domain with a magic loop."""
    for i in range(out.shape[0]):
        out[i] = a[i] * 3.0


def test_a_bare_child_launch_derives_its_work_size():
    """A bracket-less device launch is sized from the child's own magic loop.

    The same body called from the host needs no bracket either, because its loop states the
    extent. Requiring one in device code only made the two spellings of one kernel disagree.
    """
    a = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)

    @kernel
    def bare_parent(a: np.ndarray, out: np.ndarray):
        sized_child(a, out)

    bare_parent["cuda"](a, out)
    np.testing.assert_allclose(out, a * 3.0)


def test_a_bare_child_launch_and_a_bracketed_one_agree():
    """The two spellings are one launch: the derived size equals the written one."""
    a = np.arange(6, dtype=np.float32)
    derived = np.zeros(6, dtype=np.float32)
    written = np.zeros(6, dtype=np.float32)

    @kernel
    def bare_parent(a: np.ndarray, out: np.ndarray):
        sized_child(a, out)

    @kernel
    def bracketed_parent(a: np.ndarray, out: np.ndarray):
        sized_child[(out.shape[0],)](a, out)

    bare_parent["cuda"](a, derived)
    bracketed_parent["cuda"](a, written)
    np.testing.assert_allclose(derived, written)
    np.testing.assert_allclose(derived, a * 3.0)


def test_a_device_selector_in_a_child_launch_is_ignored_with_a_warning():
    """The device is already running, so a selector there cannot choose anything.

    It is *ignored*, not refused: `child["cuda"][(n,)](...)` is the spelling a reader may
    carry over from a host launch site, and the child runs wherever the parent is running
    anyway. The warning names the selector so the dead argument is not silent.
    """
    from pygpgpu.kernel.translator.nested_launch import NestedLaunchWarning

    @kernel
    def parent_with_selector(out: np.ndarray):
        child["cuda"][(out.shape[0],)](out)

    out = np.zeros(8, dtype=np.int32)
    with pytest.warns(NestedLaunchWarning, match="no effect on the child"):
        parent_with_selector["cuda"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


def test_a_plain_child_launch_warns_nothing():
    """The supported spelling stays silent: the warning is only about a dead selector."""
    from pygpgpu.kernel.translator.nested_launch import NestedLaunchWarning

    out = np.zeros(8, dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NestedLaunchWarning)
        parent["cuda"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


def test_a_selector_without_a_work_size_is_refused():
    """Ignoring the selector leaves no geometry at all, so that case still has to be named."""

    @kernel
    def selector_only(out: np.ndarray):
        child["cuda"](out)

    with pytest.raises(SyntaxError) as caught:
        selector_only["cuda"](np.zeros(4, dtype=np.int32))
    assert "work size" in str(caught.value)


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


def test_a_nested_launch_ignores_a_backend_selector_with_a_warning():
    """The device is already running, so a selector there is ignored rather than honoured.

    It used to be refused. Ignoring it (and warning) is the friendlier reading: the child
    runs wherever the parent runs, so the dead argument costs nothing, and a reader who
    carried the spelling over from a host launch site gets a warning that names it.
    """
    from pygpgpu.kernel.translator.nested_launch import NestedLaunchWarning

    @kernel
    def picks_a_backend(out: np.ndarray):
        child["opencl"][(out.shape[0],)](out)

    out = np.zeros(8, dtype=np.int32)
    with pytest.warns(NestedLaunchWarning, match="cannot|no effect"):
        picks_a_backend["cuda"](out)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.int32) * 2)


@kernel
def repeated_phases(a: np.ndarray, b: np.ndarray):
    """Two passes, so the generated file holds two kernels of this body."""
    for i in range(a.shape[0]):
        a[i] = a[i] * 2
    for i in range(a.shape[0]):
        b[i] = a[i] + 1


class PhaseMethod:

    def __init__(self) -> None:
        self.height: int = 2
        self.width: int = 2
        self.scale: float = 2.0
        self.a = np.zeros((2, 2), dtype=np.float32)
        self.b = np.zeros((2, 2), dtype=np.float32)

    def scale_of(self, i: int, j: int) -> float:
        return self.scale

    @kernel
    def draw(self):
        for i, j in np.ndindex(self.height, self.width):
            self.a[i, j] = float(i) + float(j)
        for i, j in np.ndindex(self.height, self.width):
            self.b[i, j] = self.a[i, j] * self.scale_of(i, j)


def test_a_method_with_two_magic_loops_runs_both_passes(gpu_backend):
    """A method's passes read the receiver structure and each other's fields.

    The receiver travels as a structure parameter, and a multi-pass body is one kernel
    per pass: before the phases shared the receiver signature, a method body with a second
    loop failed the argument-count check, and the launch read the loop extents from the
    wrong translator.
    """
    view = PhaseMethod()
    view.draw[gpu_backend]()
    np.testing.assert_array_equal(
        view.b, np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32))


def test_a_recompile_does_not_duplicate_the_later_passes(gpu_backend):
    """Re-rendering a multi-pass body must refresh its pieces, not append extra copies.

    The source writer merges one piece per definition, keyed by the signature line, so a
    multi-phase body has to be written as one piece per pass. Joining the passes into one
    piece made the merge replace only the first pass's group and leave the file's later
    passes in place; the file then held those passes twice and the next compile refused it
    ("function ... has already been defined"), which is what a second run of this test file
    used to hit.
    """
    values = np.arange(4, dtype=np.float32)
    result = np.zeros(4, dtype=np.float32)
    repeated_phases[gpu_backend](values, result)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))

    repeated_phases._cache.clear()  # force a full re-render of both passes
    values = np.arange(4, dtype=np.float32)
    result = np.zeros(4, dtype=np.float32)
    repeated_phases[gpu_backend](values, result)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
