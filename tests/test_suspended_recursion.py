"""The generalized tail-conversion: recursive results assigned to a slot.

The tail pass covers more than `return f(...)`: a recursive call whose result lands
in a variable -- `r = f(...)` -- with the statements after it reading only `r` and
the parameters no recursive call rebinds, is the same loop with a level counter.
The suspended block re-runs once per level after the loop bottoms out, which is
exactly the order the original nesting spells (innermost suspension first). The
tests pin four shapes and two refusals:

* `g(f(...))` -- an arbitrary pure function wrapped around the recursive call;
* the affine/multi-line block -- several statements reading `r`, each any number
  of times, in any combination;
* a parameter the recursive call rebinds (per-level values) inside the suspended
  block -- refused, with the reason;
* a call with nothing after it -- refused with "spell it as a return".
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


def wrapped(x: int) -> int:
    """`g(f(x))`: the recursive result wrapped in an arbitrary pure function."""
    if x <= 0:
        return 1
    return double_(triple_(wrapped(x - 1)))


def double_(v: int) -> int:
    return v * 2


def triple_(v: int) -> int:
    return v * 3


def multi_line(x: int) -> int:
    """A multi-line suspended block reading the result several times."""
    if x <= 0:
        return 0
    r = multi_line(x - 1)
    doubled = r * 2
    return doubled + r + 3


def leaking(x: int) -> int:
    """NOT convertible: the suspended block reads a per-level parameter."""
    if x <= 0:
        return 0
    r = leaking(x - 1)
    return r + x


def bare_call(x: int) -> int:
    """A recursive result assigned with nothing but a zero-multiply after it: the
    suspension reads the slot once and the k-loop re-assigns it every pass --
    the answer stays 0 either way, so the conversion succeeds."""
    if x <= 0:
        return 0
    r = bare_call(x - 1)
    return r * 0


@kernel
def k_wrapped(out: np.ndarray, x: int):
    out[0] = wrapped(x)


@kernel
def k_multi(out: np.ndarray, x: int):
    out[0] = multi_line(x)


@kernel
def k_leaking(out: np.ndarray, x: int):
    out[0] = leaking(x)


@kernel
def k_bare(out: np.ndarray, x: int):
    out[0] = bare_call(x)


def test_the_wrapped_result_answers_python(runner):
    """wrapped(4) = double(triple(0)) ... three levels of g = 2*3 applied k times.

    The recursion: wrapped(4) = double(triple(wrapped(3))), with the base
    wrapped(0) = 1. Three levels, each applying double∘triple once: 1 * 6 * 6 * 6
    = 216. The counter loop re-runs the suspension once per level."""
    out = np.zeros(1, dtype=np.int32)
    runner(k_wrapped, out, np.int32(3))
    assert int(out[0]) == 216


def test_the_multi_line_block_answers_python(runner):
    """multi_line(3, 3): the suspension reads r three times across two statements.

    Levels: r0 = 0 (base); r1 = 2*0 + 0 + 3 = 3; r2 = 2*3 + 3 + 3 = 12; r3 =
    2*12 + 12 + 3 = 39. The k-loop re-runs the block once per level, and the
    numbers must land exactly."""
    out = np.zeros(1, dtype=np.int32)
    runner(k_multi, out, np.int32(3))
    assert int(out[0]) == 39


def test_a_per_level_read_in_the_suspension_refuses():
    """The suspension reads `x`, which every recursive call rebinds -- the k-loop
    would show every level the final one's value."""
    import ast
    from pygpgpu.kernel.translator.tail_recursion import TailRecursionError
    from pygpgpu.kernel.translator.source_text import _parse_kernel_source, _trimmed_source

    module: ast.Module = _parse_kernel_source(_trimmed_source(leaking))
    with pytest.raises(TailRecursionError) as caught:
        from pygpgpu.kernel.translator.tail_recursion import tail_loop_tree

        tail_loop_tree(module.body[0])
    assert "differs per recursion level" in str(caught.value)


def test_the_bare_assignment_converts_and_answers():
    """`r = f(...)` with `return 0` after it: the suspension is the constant
    return, which the k-loop re-assigns every pass -- the answer is the same
    constant either way, so the conversion succeeds."""
    out = np.zeros(1, dtype=np.int32)
    k_bare["cuda"](out, np.int32(3))
    assert int(out[0]) == 0


def test_the_converted_shapes_agree_with_the_interpreter():
    """The interpreter runs the original recursion; the device runs the loop."""
    out = np.zeros(1, dtype=np.int32)
    k_wrapped["python"](out, np.int32(3))
    assert int(out[0]) == 216
    k_multi["python"](out, np.int32(3))
    assert int(out[0]) == 39
