"""A second name for array memory: aliases, slices, and from-the-end indices.

Three spellings python programmers reach for without thinking, and what they mean on a
device:

    src = a            the same buffer, so a pointer copy (`double* src = a;`)

    part = a[lo:hi]    a pointer at an offset, with a length of its own

    a[-1]              python counts from the end; a device offset does not

The last one is the reason this file exists. A device index is a plain offset, so a raw
`-1` reached the word *before* the buffer: reading `a[-1]` answered `0.0` where python
answers the last element, and `a[-1] = v` wrote outside the array -- a wrong answer and a
buffer overrun, neither of them reported. The first two were refused outright, the alias
with an `IndexError` from the launch's shape check and the slice as an unsupported
expression, so the natural spelling of "work on part of this array" did not compile.

Every test compares against what python does with the same body, because the interpreter
backend runs exactly that.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


# ---------------------------------------------------------------------------
# Aliases: `src = a`
# ---------------------------------------------------------------------------


@kernel
def read_through_alias(a: np.ndarray, out: np.ndarray):
    """The shape from the original bug: alias the array, then loop over the alias."""
    src = a
    for i in range(src.shape[0]):
        out[i] = src[i] + 1.0


@kernel
def write_through_alias(a: np.ndarray):
    """A write through the alias lands in the array, exactly as python's name sharing does."""
    src = a
    for i in range(src.shape[0]):
        src[i] = src[i] * 10.0


@kernel
def chained_alias(a: np.ndarray, out: np.ndarray):
    """Two hops: `two = one`, and `one` is itself an alias."""
    one = a
    two = one
    for i in range(two.shape[0]):
        out[i] = two[i] - 1.0


@kernel
def alias_enumerated(a: np.ndarray, out: np.ndarray):
    """`enumerate(src)`: the alias names the dispatch domain too."""
    src = a
    for i, value in enumerate(src):
        out[i] = value * 2.0


@kernel
def alias_in_ndindex(grid: np.ndarray, out: np.ndarray):
    """`np.ndindex(g.shape[...])`: the alias carries the extents as well."""
    g = grid
    for i, j in np.ndindex(g.shape[0], g.shape[1]):
        out[i, j] = g[i, j] + 1.0


@kernel
def alias_before_the_loop(a: np.ndarray, out: np.ndarray):
    """The alias is bound *above* the loop, where the host also runs the statement."""
    src = a
    for i in range(a.shape[0]):
        out[i] = src[i]


class Holder:
    """A receiver whose array member is aliased: `buf = self.data`."""

    def __init__(self):
        self.data = np.arange(4, dtype=np.float32)
        self.out = np.zeros(4, np.float32)

    @kernel
    def read_member_alias(self):
        buf = self.data
        for i in range(self.out.shape[0]):
            self.out[i] = buf[i] + 100.0


def test_a_local_alias_reads_the_same_buffer(runner):
    """`src = a` then `src[i]`, with the loop bound taken from the alias."""
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(read_through_alias, a, out)
    np.testing.assert_allclose(out, a + 1.0)


def test_a_write_through_an_alias_reaches_the_array(runner):
    """The alias is not a copy: the caller's array comes back changed."""
    a = np.arange(1, 5, dtype=np.float32)
    runner(write_through_alias, a)
    np.testing.assert_allclose(a, np.arange(1, 5, dtype=np.float32) * 10.0)


def test_an_alias_of_an_alias_resolves(runner):
    """The chain follows to the array, and its extent is the one the loop uses."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(chained_alias, a, out)
    np.testing.assert_allclose(out, a - 1.0)


def test_an_alias_can_be_the_iteration_domain(runner):
    """`enumerate(src)`: recognising the alias is what keeps this the dispatch loop."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(alias_enumerated, a, out)
    np.testing.assert_allclose(out, a * 2.0)


def test_an_alias_carries_the_extents_of_an_ndindex_domain(runner):
    """`np.ndindex(g.shape[0], g.shape[1])` over the alias."""
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros((2, 3), np.float32)
    runner(alias_in_ndindex, grid, out)
    np.testing.assert_allclose(out, grid + 1.0)


def test_an_alias_bound_above_the_loop_is_device_code_not_a_host_value(runner):
    """The statement sits in a host region, but the device still binds the pointer.

    Treating it as a host value made the alias a kernel parameter of its own, and the
    launch then looked for an extent the argument list does not carry.
    """
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(alias_before_the_loop, a, out)
    np.testing.assert_allclose(out, a)


def test_an_alias_of_a_receiver_member_reads_it(runner):
    """`buf = self.data`: the member's own parameter is what the alias points at."""
    holder = Holder()
    runner(Holder.read_member_alias, holder)
    np.testing.assert_allclose(holder.out, holder.data + 100.0)


# ---------------------------------------------------------------------------
# Slices: `part = a[lo:hi]`
# ---------------------------------------------------------------------------


@kernel
def slice_from_the_front(a: np.ndarray, out: np.ndarray):
    """`a[0:4]`: both bounds given."""
    part = a[0:4]
    for i in range(out.shape[0]):
        out[i] = part[i]


@kernel
def slice_from_an_offset(a: np.ndarray, out: np.ndarray):
    """`a[2:]`: no upper bound, so the length comes from the array's extent."""
    part = a[2:]
    for i in range(out.shape[0]):
        out[i] = part[i]


@kernel
def slice_of_a_slice(a: np.ndarray, out: np.ndarray):
    """`inner = part[1:3]`: the offsets have to compose, not restart."""
    part = a[2:6]
    inner = part[1:3]
    for i in range(inner.shape[0]):
        out[i] = inner[i]


@kernel
def slice_of_a_slice_with_an_open_bound(a: np.ndarray, out: np.ndarray):
    """`part[1:]`: the omitted bound is the *parent's* length, not the array's."""
    part = a[1:5]
    tail = part[1:]
    for i in range(tail.shape[0]):
        out[i] = tail[i]


@kernel
def slice_with_a_computed_bound(a: np.ndarray, out: np.ndarray):
    """`a[n // 2:]`: the bound is an expression, and the pointer just carries it."""
    n = a.shape[0]
    part = a[n // 2:]
    for i in range(out.shape[0]):
        out[i] = part[i]


@kernel
def slice_with_only_an_upper_bound(a: np.ndarray, out: np.ndarray):
    """`a[:3]`: no lower bound, so the slice starts at the buffer."""
    part = a[:3]
    for i in range(out.shape[0]):
        out[i] = part[i]


@kernel
def slice_length_is_the_local_shape(a: np.ndarray, out: np.ndarray):
    """`part.shape[0]` is the slice's *length*, not the array's."""
    part = a[1:4]
    for i in range(part.shape[0]):
        out[i] = part[i]


@kernel
def write_through_a_slice(a: np.ndarray):
    """A store through the slice lands in the array it views."""
    part = a[1:3]
    for i in range(part.shape[0]):
        part[i] = 0.0


@kernel
def slice_in_an_expression(a: np.ndarray, out: np.ndarray):
    """`a[2:][i]`: the slice indexed directly, with no local to hold it."""
    for i in range(out.shape[0]):
        out[i] = a[2:][i]


def test_a_slice_reads_the_elements_it_names(runner):
    """`a[0:4]`: the pointer starts at the buffer."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(slice_from_the_front, a, out)
    np.testing.assert_allclose(out, a[0:4])


def test_a_slice_at_an_offset_reads_from_there(runner):
    """`a[2:]`: the pointer starts at the offset the slice names."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(slice_from_an_offset, a, out)
    np.testing.assert_allclose(out, a[2:])


def test_an_open_lower_bound_starts_at_the_buffer(runner):
    """`a[:3]` is the same pointer with a length, not a new array."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(3, np.float32)
    runner(slice_with_only_an_upper_bound, a, out)
    np.testing.assert_allclose(out, a[:3])


def test_a_slices_shape_is_its_own_length(runner):
    """`a[1:4]` iterates three elements, not six: the local length is the slice's."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(3, np.float32)
    runner(slice_length_is_the_local_shape, a, out)
    np.testing.assert_allclose(out, a[1:4])


def test_a_write_through_a_slice_reaches_the_array(runner):
    """A slice is a view, so the elements it names are the array's own."""
    a = np.arange(6, dtype=np.float32)
    runner(write_through_a_slice, a)
    expected = np.arange(6, dtype=np.float32)
    expected[1:3] = 0.0
    np.testing.assert_allclose(a, expected)


def test_a_slice_indexed_directly_needs_no_local(runner):
    """`a[2:][i]` folds the offset into the subscript."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(slice_in_an_expression, a, out)
    np.testing.assert_allclose(out, a[2:])


def test_a_slice_of_a_slice_composes_its_offsets(runner):
    """`a[2:6][1:3]` starts where the parent slice does, not where the array does."""
    a = np.arange(1, 8, dtype=np.float32)
    out = np.zeros(2, np.float32)
    runner(slice_of_a_slice, a, out)
    np.testing.assert_allclose(out, a[2:6][1:3])


def test_an_open_bound_on_a_slice_of_a_slice_is_the_parents_length(runner):
    """`a[1:5][1:]` is three elements: the parent's length minus one, not the array's."""
    a = np.arange(1, 8, dtype=np.float32)
    out = np.zeros(3, np.float32)
    runner(slice_of_a_slice_with_an_open_bound, a, out)
    np.testing.assert_allclose(out, a[1:5][1:])


def test_a_slice_bound_may_be_an_expression(runner):
    """`a[n // 2:]` is a view like any other; the pointer carries the computed start."""
    a = np.arange(6, dtype=np.float32)
    out = np.zeros(3, np.float32)
    runner(slice_with_a_computed_bound, a, out)
    np.testing.assert_allclose(out, a[a.shape[0] // 2:])


def test_a_reversed_slice_is_refused_with_the_reason(gpu_backend):
    """`a[::-1]` is not a pointer: a work item cannot walk a buffer backwards.

    The refusal names the spelling that does work, because the wrong answer here would be
    silent -- the slice still addresses the same buffer, just not in that order.
    """

    @kernel
    def reversed_slice(a: np.ndarray, out: np.ndarray):
        rev = a[::-1]
        for i in range(out.shape[0]):
            out[i] = rev[i]

    a = np.arange(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as refused:
        reversed_slice[gpu_backend](a, np.zeros(4, np.float32))
    message = str(refused.value)
    assert "reversed" in message or "step" in message
    assert "a[n - 1 - i]" in message


def test_a_slice_of_a_multidimensional_array_is_refused(gpu_backend):
    """On a 2-D array a slice is whole *rows*, and a device pointer addresses elements.

    `part = grid[0:2]` then `part[i]` is a row in python. Emitting the offset would answer
    the scalar `grid[0 + i]` instead -- a silent scalar where python has an array. The rank
    is known only at the call, so the refusal happens there.
    """

    @kernel
    def slice_rows(grid: np.ndarray, out: np.ndarray):
        part = grid[0:2]
        for i in range(out.shape[0]):
            out[i] = part[i]

    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    with pytest.raises(ValueError) as refused:
        slice_rows[gpu_backend](grid, np.zeros(2, np.float32))
    message = str(refused.value)
    assert "slice" in message
    assert "elements" in message


# ---------------------------------------------------------------------------
# Negative indices: `a[-1]`
# ---------------------------------------------------------------------------


@kernel
def last_element(a: np.ndarray, out: np.ndarray):
    """`a[-1]`: python's last element."""
    for i in range(out.shape[0]):
        out[i] = a[-1]


@kernel
def second_to_last_element(a: np.ndarray, out: np.ndarray):
    """`a[-2]`: the offset is counted from the extent, not from zero."""
    for i in range(out.shape[0]):
        out[i] = a[-2]


@kernel
def computed_index(a: np.ndarray, k: int, out: np.ndarray):
    """An index that arrives as a value: passed through, as it always was."""
    for i in range(out.shape[0]):
        out[i] = a[k]


@kernel
def write_to_the_last_element(a: np.ndarray):
    """`a[-1] = v`: the store has to land inside the array, not before it."""
    for i in range(a.shape[0]):
        a[-1] = 99.0


@kernel
def last_element_of_a_grid(grid: np.ndarray, out: np.ndarray):
    """A tuple subscript with a negative in each axis."""
    for i, j in np.ndindex(grid.shape[0], grid.shape[1]):
        out[i, j] = grid[-1, -1]


@kernel
def alias_negative_index(a: np.ndarray, out: np.ndarray):
    """A negative index through an alias: the extent is the array's, not the alias's."""
    src = a
    for i in range(out.shape[0]):
        out[i] = src[-1]


def test_last_element_is_the_arrays_last(runner):
    """`a[-1]` used to read the word *before* the buffer and answer 0.0."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(last_element, a, out)
    np.testing.assert_allclose(out, np.full(4, a[-1], np.float32))


def test_a_further_negative_offset_counts_from_the_extent(runner):
    """`a[-2]` is `extent - 2`, which is the second-to-last element."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(second_to_last_element, a, out)
    np.testing.assert_allclose(out, np.full(4, a[-2], np.float32))


def test_a_computed_index_is_passed_through_unchanged(runner):
    """A value index is emitted as it is written -- no sign test around every access.

    Wrapping it would evaluate the expression twice (a call or a subscript would run
    twice) and would put a select around every array access in every kernel. A computed
    index that goes negative is out of range by python's own rules, which is the same kind
    of mistake as indexing past the end -- and the device has never bounds-checked either.
    """
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(computed_index, a, np.int32(2), out)
    np.testing.assert_allclose(out, np.full(4, a[2], np.float32))


def test_a_folded_negative_index_reads_the_same_element_as_the_explicit_spelling(runner):
    """`a[-1]` and `a[n - 1]` are one access, so the fold is not a special case."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(last_element, a, out)
    np.testing.assert_allclose(out, np.full(4, a[a.shape[0] - 1], np.float32))


def test_a_write_to_the_last_element_stays_inside(runner):
    """`a[-1] = v` wrote outside the array; the last element is where it belongs."""
    a = np.arange(1, 5, dtype=np.float32)
    runner(write_to_the_last_element, a)
    expected = np.arange(1, 5, dtype=np.float32)
    expected[-1] = 99.0
    np.testing.assert_allclose(a, expected)


def test_a_negative_index_in_a_tuple_subscript_counts_per_axis(runner):
    """Each axis is corrected against its own extent."""
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.zeros((2, 3), np.float32)
    runner(last_element_of_a_grid, grid, out)
    np.testing.assert_allclose(out, np.full((2, 3), grid[-1, -1], np.float32))


def test_a_negative_index_through_an_alias_uses_the_arrays_extent(runner):
    """The alias has no extent of its own, so the correction resolves to the array's."""
    a = np.arange(1, 5, dtype=np.float32)
    out = np.zeros(4, np.float32)
    runner(alias_negative_index, a, out)
    np.testing.assert_allclose(out, np.full(4, a[-1], np.float32))


def test_a_negative_index_into_a_shared_buffer_counts_from_its_axis(gpu_backend):
    """A `local_array_per_group` buffer has a compile-time shape, so its axes are known.

    Left raw, `buf[blockIdx][-1]` addressed the shared array's own header -- an illegal
    address rather than a wrong number, because shared memory has nothing before it.
    """
    from pygpgpu import barrier, blockIdx, local_array_per_group

    @kernel
    def shared_last(out: np.ndarray):
        buf = local_array_per_group(4, np.float32)
        for i in range(out.shape[0]):
            buf[blockIdx][i] = i
            barrier()
            out[i] = buf[blockIdx][-1]

    out = np.zeros(4, np.float32)
    shared_last[gpu_backend](out)
    np.testing.assert_allclose(out, np.full(4, 3.0))
