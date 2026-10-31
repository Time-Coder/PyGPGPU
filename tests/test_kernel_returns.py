"""Feature 1: return values of a @kernel function.

A kernel runs once per work item, so a C return value has no single meaning. The
translator therefore gives every returned value a device parameter of its own (a
one-element buffer for a scalar or vector, the array object itself for an array) and
lets the first work item that reaches a `return` claim it. The host reads the buffers
back after the launch and rebuilds the Python result: a single value, a tuple, or None
when no work item returned anything.

The python backend runs the body as written, so its result is whatever the function
itself returns; the two must agree, which is what these tests check.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from pygpgpu import kernel, get_global_id, work_size
from pygpgpu.vec_types import float4


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@kernel
def no_return(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i]


@kernel
def return_float(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return x[i]


@kernel
def return_int(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return 42


@kernel
def return_bool(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return x[i] > 0.0


@kernel
def return_after_accumulating(x: np.ndarray):
    total = 0.0
    for i in range(x.shape[0]):
        total = total + x[i]
        if i == 0:
            return total


@kernel
def return_tuple(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return x[i], 7


@kernel
def return_vector(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return float4(x[i], x[i] * 2.0, x[i] * 3.0, x[i] * 4.0)


@kernel
def return_bare(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return


@kernel
def return_none(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return None


@kernel
def return_array_param(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * 3.0
    return out


@kernel
def return_from_nested(x: np.ndarray):
    """`return` from a nested loop and a nested if, not from the outermost statement."""
    for i in range(x.shape[0]):
        for j in range(2):
            if j == 1:
                if i == 0:
                    return x[i] + j


@kernel
def return_same_value_from_every_item(x: np.ndarray):
    """Every work item returns the same value, so "first one wins" is still exact."""
    for i in range(x.shape[0]):
        return x[i] + x[i] * 0.0


@kernel
def scalar_argument_return(x: np.ndarray, factor: float):
    for i in range(x.shape[0]):
        if i == 0:
            return x[i] * factor


@kernel
def gid_style_return(x: np.ndarray):
    work_size(x)
    i = get_global_id(0)
    if i == 0:
        return x[i] * 10.0


@kernel
def return_called_from_helper(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return square(x[i])


def square(value):
    return value * value


# --- differing return sites -------------------------------------------------
#
# Python allows each `return` its own shape, and the implementations must follow: the
# winning site is recorded so the host rebuilds exactly that site's value. These kernels
# return different things from different branches, and each branch has a test that makes
# only that branch run, so both shapes are exercised.


@kernel
def two_shapes_by_branch(x: np.ndarray, flag: bool):
    for i in range(x.shape[0]):
        if i == 0:
            if flag:
                return x[i]
            return x[i], 1


@kernel
def value_or_none_by_branch(x: np.ndarray, flag: bool):
    for i in range(x.shape[0]):
        if i == 0:
            if flag:
                return x[i]
            return


@kernel
def int_then_float_sites(x: np.ndarray, flag: bool):
    for i in range(x.shape[0]):
        if i == 0:
            if flag:
                return 3
            return x[i]


@kernel
def array_or_value_by_branch(x: np.ndarray, out: np.ndarray, flag: bool):
    for i in range(out.shape[0]):
        out[i] = x[i]
        if i == 0:
            if flag:
                return out
            return x[i]


# --- translation-stage rejections -----------------------------------------


@kernel
def nested_tuple_return(x: np.ndarray):
    for i in range(x.shape[0]):
        if i == 0:
            return (x[i], x[i]), 1


@kernel
def return_array_expression(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        return out[i:]


@kernel
def return_after_magic_loop(x: np.ndarray):
    """A value-returning return after the magic loop cannot mean the same as in python."""
    total = 0.0
    for i in range(x.shape[0]):
        total = total + x[i]
    return total

@kernel
def assignment_return(x: np.ndarray):
    for i in range(x.shape[0]):
        return i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_return_is_none(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    assert runner(no_return, x, out) is None
    np.testing.assert_allclose(out, x)


def test_float_return(runner):
    x = np.array([2.5, 9.0, 1.0], dtype=np.float32)
    assert float(runner(return_float, x)) == 2.5


def test_int_return(runner):
    x = np.arange(3, dtype=np.float32)
    assert int(runner(return_int, x)) == 42


def test_bool_return(runner):
    # The python backend hands back whatever numpy produced (np.bool_), while a device
    # return is rebuilt from its byte slot as a python bool; both mean the same value.
    x = np.array([1.5, 0.0], dtype=np.float32)
    assert bool(runner(return_bool, x)) is True

def test_return_after_accumulating(runner):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert float(runner(return_after_accumulating, x)) == 1.0


def test_tuple_return(runner):
    x = np.array([4.0, 5.0], dtype=np.float32)
    first, second = runner(return_tuple, x)
    assert float(first) == 4.0
    assert int(second) == 7


def test_vector_return(runner):
    x = np.array([1.5], dtype=np.float32)
    value = runner(return_vector, x)
    assert isinstance(value, float4)
    assert (value.x, value.y, value.z, value.w) == (1.5, 3.0, 4.5, 6.0)


def test_bare_return_is_none(runner):
    x = np.arange(3, dtype=np.float32)
    assert runner(return_bare, x) is None


def test_explicit_none_return_is_none(runner):
    x = np.arange(3, dtype=np.float32)
    assert runner(return_none, x) is None


def test_array_return_is_the_same_object(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    result = runner(return_array_param, x, out)
    assert result is out
    np.testing.assert_allclose(out, x * 3.0)


def test_return_from_nested_statement(runner):
    x = np.array([6.0, 7.0], dtype=np.float32)
    assert float(runner(return_from_nested, x)) == 7.0


def test_every_item_returning_the_same_value(runner):
    x = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    assert float(runner(return_same_value_from_every_item, x)) == 3.0


def test_return_of_a_scalar_argument(runner):
    x = np.array([8.0, 2.0], dtype=np.float32)
    assert float(runner(scalar_argument_return, x, np.float32(0.25))) == 2.0


def test_return_of_a_helper_call(runner):
    x = np.array([3.0, 4.0], dtype=np.float32)
    assert float(runner(return_called_from_helper, x)) == 9.0


def test_gid_style_return(gpu_backend, runner):
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    assert float(runner(gid_style_return, x)) == 5.0


def test_submit_returns_the_value(backend):
    x = np.array([3.25, 8.0], dtype=np.float32)
    target = return_float["python"] if backend == "python" else return_float[backend]
    assert float(target.submit(x).result(timeout=120)) == 3.25


def test_async_call_returns_the_value(backend):
    x = np.array([4.75, 8.0], dtype=np.float32)
    target = return_float["python"] if backend == "python" else return_float[backend]
    assert float(asyncio.run(_await_launch(target, x))) == 4.75


async def _await_launch(target, x):
    """Launch from inside the loop, which is how async_call is meant to be called."""
    return await target.async_call(x)


def test_different_shapes_by_branch_are_supported(runner):
    """A value from one branch and a tuple from another: the branch that wins decides."""
    x = np.array([5.0, 6.0], dtype=np.float32)
    assert float(runner(two_shapes_by_branch, x, True)) == 5.0
    first, second = runner(two_shapes_by_branch, x, False)
    assert float(first) == 5.0 and int(second) == 1


def test_value_or_none_by_branch(runner):
    x = np.array([2.0, 3.0], dtype=np.float32)
    assert float(runner(value_or_none_by_branch, x, True)) == 2.0
    assert runner(value_or_none_by_branch, x, False) is None


def test_two_scalar_sites_of_different_types(runner):
    """An int slot and a float slot coexist: each site writes its own type's slot."""
    x = np.array([2.5, 3.0], dtype=np.float32)
    assert int(runner(int_then_float_sites, x, True)) == 3
    assert float(runner(int_then_float_sites, x, False)) == 2.5


def test_array_or_value_by_branch(runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    assert runner(array_or_value_by_branch, x, out, True) is out
    assert float(runner(array_or_value_by_branch, x, out, False)) == 0.0

def test_return_after_the_magic_loop_is_rejected(gpu_backend, runner):
    """A per-work-item value cannot be returned after the loop; a shared one can.

    `return_after_magic_loop` accumulates with a plain `total = total + x[i]`, so each work
    item would compute its own partial sum and there is no single value to claim. The error
    names that race rather than the return, because the *store* is what is wrong -- the
    `+=` spelling makes it one shared slot and the return legal (see
    `test_a_scalar_written_in_the_loop_is_one_location`).
    """
    with pytest.raises(SyntaxError) as caught:
        runner(return_after_magic_loop, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    message = str(caught.value)
    assert "shared by every work item" in message
    assert "+=" in message


@pytest.mark.parametrize(
    "kernel_func, args",
    [
        (nested_tuple_return, (np.arange(2, dtype=np.float32),)),
    ],
)
def test_unsupported_returns_are_rejected_at_translation(gpu_backend, runner, kernel_func, args):
    with pytest.raises(SyntaxError):
        runner(kernel_func, *args)


def test_nested_tuple_message_explains_the_shape(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(nested_tuple_return, np.arange(2, dtype=np.float32))
    assert "nested" in str(caught.value)


def test_returning_an_array_expression_is_rejected(gpu_backend, runner):
    x = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError) as caught:
        runner(return_array_expression, x, out)
    assert "name" in str(caught.value)
