"""The recursive call at any argument position of the wrapping expression.

The nested-site conversion replaces the one recursive call with the slot name
wherever it sits -- the first argument of `g`, a later one, or one of several
arguments while the others carry the walk's own values. The suspension rule is
the same wherever the call sat: the remaining expression may read only the slot
and the names that keep one value for the whole walk. A parameter the call
passes *as itself* (`f(n - 1, step)`) keeps its value -- the rebinding is an
identity -- so the wrapping expression may read it, which is what lets a helper
receive the same constant at every level.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel


def combine(left: int, mid: int, right: int) -> int:
    return left * 100 + mid * 10 + right


def side_spelled(n: int, step: int) -> int:
    """The recursive call is the MIDDLE argument; `step` travels through."""
    if n <= 0:
        return 0
    return combine(step, side_spelled(n - 1, step), step)


def first_spelled(n: int, weight: int) -> int:
    """The recursive call is the FIRST argument, under a module-level helper."""
    if n <= 0:
        return 0
    return combine(apply_weight(weight, first_spelled(n - 1, weight)), 0, weight)


def apply_weight(weight: int, value: int) -> int:
    return value * weight


def two_calls(n: int, step: int) -> int:
    """NOT convertible: two recursive calls in one expression -- the counter
    cannot order two independent walks with one level count."""
    if n <= 0:
        return 0
    return two_calls(n - 1, step) + two_calls(n - 2, step)


@kernel
def k_side(out: np.ndarray, n: int, step: int):
    out[0] = side_spelled(n, step)


@kernel
def k_first(out: np.ndarray, n: int, weight: int):
    out[0] = first_spelled(n, weight)


def test_the_middle_argument_answers_python(gpu_backend):
    """side_spelled(3, 2): three levels, each combine(step, r, step) = 2*100 +
    r*10 + 2; the base 0 gives 202 -> 2222 -> 22422."""
    out = np.zeros(1, dtype=np.int32)
    k_side[gpu_backend](out, np.int32(3), np.int32(2))
    assert int(out[0]) == 22422


def test_the_first_argument_answers_python(gpu_backend):
    """first_spelled(3, 5): each level wraps the previous result once more:
    combine(5*w, 0, w) with w = 5 -- three nestings of the same helper."""
    out = np.zeros(1, dtype=np.int32)
    k_first[gpu_backend](out, np.int32(3), np.int32(5))
    expected = 0
    for _ in range(3):
        expected = combine(apply_weight(5, expected), 0, 5)
    assert int(out[0]) == expected


def test_two_calls_in_one_expression_refuse():
    """Two recursive calls in one expression: one counter cannot interleave two
    walks, and the scan refuses rather than guess an order."""
    import ast
    from pygpgpu.kernel.translator.tail_recursion import TailRecursionError, tail_loop_tree
    from pygpgpu.kernel.translator.source_text import _parse_kernel_source, _trimmed_source

    module: ast.Module = _parse_kernel_source(_trimmed_source(two_calls))
    with pytest.raises(TailRecursionError) as caught:
        tail_loop_tree(module.body[0])
    assert "twice" in str(caught.value) or "two" in str(caught.value)
