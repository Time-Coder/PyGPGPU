"""Python sequences as kernel arguments, and as structure members.

Two shapes used to have no working spelling at all, and this module pins them:

* a **bare numeric list/tuple argument** (`list[int]`, `list[float]`, a tuple, and a 2-D list)
  -- it reached the object scan, which answered "a list of objects, but the body never reads a
  member of its elements". It is an **array** argument, and `np.asarray` is what makes it one;
* a **numeric list as an object member** (`bag.values[0]`) -- it was refused with "cannot
  become a structure member yet". It becomes a **fixed-size array field** (`T * N`), which
  numpy already records as a sub-array dtype, so `structs` writes `float values[4];`.

What must NOT change is also pinned here: a list of **objects** is still an object collection
with a synthesized element structure (that is what the ray-tracing example relies on), and the
width of a converted list follows numpy's own dtype rather than the `int=`/`float=` mapping,
exactly as an ndarray argument does.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel
from pygpgpu.kernel.objects import numeric_list_as_array


# ---------------------------------------------------------------------------
# The conversion helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, dtype_name", [
    ([1, 2, 3], "int64"),
    ([1.0, 2.0], "float64"),
    ([True, False], "bool"),
    ([1 + 2j], "complex128"),
    ((1, 2, 3), "int64"),
    (np.array([1.5]), "float64"),
], ids=["list[int]", "list[float]", "list[bool]", "list[complex]", "tuple", "ndarray"])
def test_a_numeric_sequence_becomes_the_array_numpy_makes(value, dtype_name):
    converted = numeric_list_as_array(value)
    assert isinstance(converted, np.ndarray)
    assert converted.dtype.name == dtype_name


def test_a_nested_numeric_list_becomes_a_two_dimensional_array():
    converted = numeric_list_as_array([[1, 2], [3, 4]])
    assert isinstance(converted, np.ndarray)
    assert converted.shape == (2, 2)


@pytest.mark.parametrize("value", [[], (), "abc", {1, 2}, None, 3.5, [object()]],
                         ids=["empty-list", "empty-tuple", "str", "set", "None",
                              "a-number", "list-of-objects"])
def test_anything_that_is_not_a_numeric_sequence_is_left_alone(value):
    """A list of objects must stay a list: the object scan gives it a structure element."""
    assert numeric_list_as_array(value) is value


# ---------------------------------------------------------------------------
# A bare numeric sequence as a kernel argument
# ---------------------------------------------------------------------------


def test_a_list_of_int_is_an_array_argument(runner):
    seen: list[int] = []

    @kernel
    def index_it(items, out):
        for i in range(out.shape[0]):
            out[i] = items[i]

    out = np.zeros(4, dtype=np.float32)
    runner(index_it, [10, 20, 30, 40], out)
    seen.extend(out.tolist())
    assert seen == [10.0, 20.0, 30.0, 40.0]


def test_iterating_a_list_of_numbers_sums_it(runner):
    """This is the spelling that used to fail: `for x in items: total += x`."""

    @kernel
    def sum_it(items, out):
        for i in range(out.shape[0]):
            total = 0.0
            for item in items:
                total += item
            out[i] = total

    out = np.zeros(2, dtype=np.float32)
    runner(sum_it, [1.5, 2.5], out)
    assert out.tolist() == [4.0, 4.0]


def test_a_tuple_is_an_array_argument_too(runner):
    @kernel
    def index_it(items, out):
        for i in range(out.shape[0]):
            out[i] = items[i]

    out = np.zeros(3, dtype=np.float32)
    runner(index_it, (5, 6, 7), out)
    assert out.tolist() == [5.0, 6.0, 7.0]


def test_a_converted_list_keeps_numpy_width_not_the_type_map():
    """A list is an array, and an array names its own width -- `int=` must not touch it."""
    from pygpgpu.kernel.decorator import infer_ctype

    mapping = {"int": "uchar", "float": "float"}
    converted = numeric_list_as_array([1, 2, 3])
    assert infer_ctype(converted, mapping) == ("*", "long")
    assert infer_ctype(converted) == ("*", "long")


def test_a_list_of_objects_is_still_an_object_collection(runner):
    """The ray-tracing shape: an object list whose elements are read member by member."""

    class Item:
        def __init__(self, value: float) -> None:
            self.value = value

    @kernel
    def sum_members(items, out):
        for i in range(out.shape[0]):
            total = 0.0
            for item in items:
                total += item.value
            out[i] = total

    out = np.zeros(2, dtype=np.float32)
    runner(sum_members, [Item(1.0), Item(2.5)], out)
    assert out.tolist() == [3.5, 3.5]


# ---------------------------------------------------------------------------
# A numeric list as a structure member
# ---------------------------------------------------------------------------


class Bag:
    """A plain object whose member is a list of numbers."""

    def __init__(self, values: list[float]) -> None:
        self.values = values


class Mixed:
    """A list that mixes an int and a float: the members must promote to one width."""

    def __init__(self) -> None:
        self.values = [1, 2.5]


class Grid:
    """A list of lists: not an array member."""

    def __init__(self) -> None:
        self.cells = [[1, 2], [3, 4]]


class Empty:
    def __init__(self) -> None:
        self.values: list[float] = []




class WithObjects:
    def __init__(self) -> None:
        self.items = [Bag([1.0])]


def test_a_numeric_list_member_is_a_fixed_array_field(runner):
    @kernel
    def read_first(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.values[0]

    out = np.zeros(2, dtype=np.float32)
    runner(read_first, Bag([1.5, 2.5, 3.5]), out)
    assert out.tolist() == [1.5, 1.5]


def test_every_index_of_the_array_field_is_reachable(runner):
    @kernel
    def read_by_index(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.values[i]

    out = np.zeros(4, dtype=np.float32)
    runner(read_by_index, Bag([10.0, 20.0, 30.0, 40.0]), out)
    assert out.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_the_array_field_carries_the_length_the_caller_passed(runner):
    """The C layout fixes the extent, so a two-element list must read [1], not [2]."""

    @kernel
    def read_second(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.values[1]

    out = np.zeros(2, dtype=np.float32)
    runner(read_second, Bag([7.0, 9.0]), out)
    assert out.tolist() == [9.0, 9.0]


def test_a_mixed_int_and_float_list_promotes_to_one_width(runner):
    @kernel
    def read_both(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.values[0] + bag.values[1]

    out = np.zeros(2, dtype=np.float32)
    runner(read_both, Mixed(), out)
    assert out.tolist() == [3.5, 3.5]


def test_an_empty_list_member_is_refused_with_the_reason(gpu_backend):
    """The refusal belongs to the structure scan, which only a device backend runs."""

    @kernel
    def read_first(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.values[0]

    out = np.zeros(2, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        read_first[gpu_backend](Empty(), out)
    assert "empty list" in str(caught.value)


def test_a_list_of_lists_member_is_refused_with_the_reason(gpu_backend):
    @kernel
    def read_cell(grid, out):
        for i in range(out.shape[0]):
            out[i] = grid.cells[0]

    out = np.zeros(2, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        read_cell[gpu_backend](Grid(), out)
    assert "list of lists" in str(caught.value)
