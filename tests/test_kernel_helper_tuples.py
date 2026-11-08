"""`enumerate(...)` over a literal list, and a helper that answers two values.

Two spellings that read as ordinary python and had no device form:

    weights = [1.0, 2.0, 3.0]           a literal list is a compile-time table: nothing has
    for i, w in enumerate(weights):     to be built or passed, so the dispatch reads it

    lo, hi = bounds(row, column)        a helper answering a pair, unpacked at the call site

The first used to fail with "unsupported expression: List" -- while the same list written
as `np.array([...])` worked, so the rule was really "not a literal list" rather than "not a
list". The second failed with "cannot tell the type of the value this helper answers",
because the type inference only read `np.<dtype>(...)` casts and float constants out of the
helper's returns; `return n, n + 2` was arithmetic over the helper's own parameters, which
is the ordinary way to write it.

The list case is a compile-time table on purpose: the length is the dispatch bound and each
element is a constant, so no buffer is created and no argument is passed.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel
from pygpgpu.kernel.translator import FunctionTranslator
from pygpgpu.kernel.decorator import infer_ctype


def generate(func, *types, backend: str = "cuda") -> str:
    """The rendered device source for `func`, without compiling it.

    Accepts the decorated kernel or the plain function: the translator needs the source of
    the function itself.
    """
    raw = getattr(func, "raw_func", func)
    argument_types = [infer_ctype(t) for t in types]
    translator = FunctionTranslator(raw, argument_types, backend, lambda name: None, {},
                                    is_kernel=True)
    translator.translate()
    return "\n\n".join(source for _, source in translator.render_kernels("k_probe"))


# ---------------------------------------------------------------------------
# A literal list as the iteration domain
# ---------------------------------------------------------------------------


@kernel
def inline_literal_list(out: np.ndarray):
    """`enumerate([...])` written at the loop."""
    for i, w in enumerate([1.0, 2.0, 3.0, 4.0]):
        out[i] = w


@kernel
def named_literal_list(out: np.ndarray):
    """The same table bound to a name above the loop."""
    weights = [1.0, 2.0, 3.0, 4.0]
    for i, w in enumerate(weights):
        out[i] = w


@kernel
def literal_int_list(out: np.ndarray):
    """Integer elements: the table's own width is the element type."""
    steps = [10, 20, 30]
    for i, step in enumerate(steps):
        out[i] = out[i] + step


@kernel
def literal_list_used_in_a_body(out: np.ndarray):
    """The field is a value like any other: it takes part in the expression."""
    weights = [2.0, 4.0]
    for i, w in enumerate(weights):
        out[i] = w * (i + 1)


def test_an_inline_literal_list_is_the_iteration_domain(runner):
    """The dispatch covers exactly the list's length."""
    out = np.zeros(4, np.float32)
    runner(inline_literal_list, out)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])


def test_a_named_literal_list_is_the_iteration_domain(runner):
    """Binding it to a name first changes nothing: it is still a compile-time table."""
    out = np.zeros(4, np.float32)
    runner(named_literal_list, out)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])


def test_an_integer_list_keeps_its_integer_elements(runner):
    """No float appears in the table, so the element arrives as an int."""
    out = np.zeros(3, np.float32)
    runner(literal_int_list, out)
    np.testing.assert_allclose(out, [10.0, 20.0, 30.0])


def test_a_literal_element_is_a_value_in_the_body(runner):
    """The element is a normal local, so it can be used in any expression."""
    out = np.zeros(2, np.float32)
    runner(literal_list_used_in_a_body, out)
    np.testing.assert_allclose(out, [2.0, 8.0])


def test_a_literal_list_costs_no_argument_or_buffer():
    """It really is a table in the source: no parameter, and no array is built.

    That is the difference between this and `np.array([...])`, which the host builds and
    passes -- so the two spellings are the same program with different launch arguments.
    """
    source = generate(named_literal_list, np.zeros(4, np.float32))
    assert "weights" not in source
    # A table, not a buffer: the values appear as literals in the emitted code.
    assert "1.0" in source and "4.0" in source


def test_the_table_length_bounds_the_dispatch():
    """The guard is the list's own length, so no extra work item reads a missing element."""
    source = generate(named_literal_list, np.zeros(4, np.float32))
    assert ">= 4" in source
# ---------------------------------------------------------------------------
# A helper that answers two values
# ---------------------------------------------------------------------------


def bounds(row, column):
    """Two numbers from two numbers: the ordinary shape, with no cast to read."""
    return row, column


def span(start, width):
    """Arithmetic over the helper's own parameters."""
    return start, start + width


def mixed(start):
    """One integer and one float: the pair takes a type each."""
    return start, start * 1.5


def scaled(value):
    """A cast still names its width, and it wins over the inference."""
    return np.float32(value), np.int32(value * 2)


@kernel
def unpack_in_the_loop(grid: np.ndarray, out: np.ndarray):
    """`lo, hi = bounds(...)` inside the work item."""
    for i in range(out.shape[0]):
        lo, hi = bounds(i, i + 2)
        out[i] = hi - lo


@kernel
def unpack_above_the_loop(out: np.ndarray):
    """Bound above the loop, where the host is the one that calls the helper."""
    lo, hi = span(1, 2)
    for i in range(out.shape[0]):
        out[i] = hi - lo


@kernel
def unpack_a_mixed_pair(out: np.ndarray):
    """The two destinations do not have to share a type."""
    for i in range(out.shape[0]):
        whole, part = mixed(i)
        out[i] = whole + part


@kernel
def unpack_a_casted_pair(value: float, out: np.ndarray):
    """An explicit cast in the helper's return decides that element's width."""
    for i in range(out.shape[0]):
        small, doubled = scaled(value)
        out[i] = small + doubled


def test_a_helper_pair_unpacks_inside_the_loop(runner):
    """`lo, hi = bounds(...)`: the pair travels by address and lands in two locals."""
    grid = np.zeros(4, np.float32)
    out = np.zeros(4, np.float32)
    runner(unpack_in_the_loop, grid, out)
    np.testing.assert_allclose(out, np.full(4, 2.0))


def test_a_helper_pair_unpacks_above_the_loop(runner):
    """The host runs the helper, so the device receives two scalars."""
    out = np.zeros(4, np.float32)
    runner(unpack_above_the_loop, out)
    np.testing.assert_allclose(out, np.full(4, 2.0))


def test_the_two_destinations_may_have_different_types(runner):
    """One int and one float, each with its own declared width."""
    out = np.zeros(4, np.float32)
    runner(unpack_a_mixed_pair, out)
    np.testing.assert_allclose(out, [i + i * 1.5 for i in range(4)])


def test_a_cast_in_the_helper_decides_the_width(runner):
    """`np.float32(value), np.int32(...)`: the casts are what the inference reads first."""
    out = np.zeros(4, np.float32)
    runner(unpack_a_casted_pair, np.float32(2.0), out)
    np.testing.assert_allclose(out, np.full(4, 6.0))


def table_lookup(value):
    """A helper only the interpreter can run: a dict has no device form."""
    table = {1: 10, 2: 20}
    return table[value], table[value] * 2


@kernel
def unpack_from_a_host_only_helper(out: np.ndarray):
    """The pair comes from a helper the device could not translate at all."""
    lo, hi = table_lookup(1)
    for i in range(out.shape[0]):
        out[i] = hi - lo


def test_the_helper_is_not_called_on_the_device_when_it_is_unpacked_above(runner):
    """Above the loop the pair is *data*, so the helper needs no device meaning.

    `table_lookup` reads a dict, which the device has no form for -- so a body that used it
    as a device helper could not be translated at all. That it runs here is the proof that
    the host is the one calling it.
    """
    out = np.zeros(4, np.float32)
    runner(unpack_from_a_host_only_helper, out)
    np.testing.assert_allclose(out, np.full(4, 10.0))
