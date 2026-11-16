"""Black-box probes, round two: the shapes a caller reaches for after the basics work.

Async launching, structure arguments, tuple returns, dtypes that are not float32, and the
"write then read the same array" pattern. Each probe is the usage a reader would try; the
ones that answered wrongly became regression tests.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from pygpgpu import kernel

# ---------------------------------------------------------------------------
# Async: the value the future carries must be the value the call returns
# ---------------------------------------------------------------------------


@kernel
def add_one(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] + 1.0


def test_awaiting_an_async_call_answers_the_written_array(gpu_backend):
    """`await k.async_call(...)` must not resolve before the read-back has landed."""
    import asyncio

    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)

    async def run():
        await add_one[gpu_backend].async_call(a, out)

    asyncio.run(run())
    np.testing.assert_array_equal(out, a + 1.0)


def test_submit_then_result_answers_the_written_array(gpu_backend):
    """`k.submit(...).result()` -- a plain concurrent future -- same guarantee."""
    a = np.arange(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    add_one[gpu_backend].submit(a, out).result()
    np.testing.assert_array_equal(out, a + 1.0)


# ---------------------------------------------------------------------------
# A tuple return from a body that returns several positions
# ---------------------------------------------------------------------------


@kernel
def two_results(a: np.ndarray):
    """A tuple return: python answers `(a[0], a[-1])`-shaped data per work item."""
    for i in range(a.shape[0]):
        return a[i], a[i] * 2.0
    return 0.0, 0.0


def test_a_tuple_return_comes_back_as_a_tuple(gpu_backend):
    a = np.array([3.0, 4.0], dtype=np.float32)
    result = two_results[gpu_backend][(1, 2)](a)
    assert isinstance(result, tuple) and len(result) == 2
    assert float(result[0]) == pytest.approx(3.0)
    assert float(result[1]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Integer and double dtypes, not just float32
# ---------------------------------------------------------------------------


@kernel
def sum_ints(a: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = a[i] * 2


def test_an_int32_array_round_trips(gpu_backend):
    a = np.array([1, 2, 3], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)
    sum_ints[gpu_backend](a, out)
    np.testing.assert_array_equal(out, np.array([2, 4, 6], dtype=np.int32))


def test_a_float64_array_round_trips(gpu_backend):
    a = np.array([1.5, 2.5], dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    add_one[gpu_backend](a, out)
    np.testing.assert_array_equal(out, np.array([2.5, 3.5], dtype=np.float64))


# ---------------------------------------------------------------------------
# A structure argument: the caller's ctypes value comes back with its writes
# ---------------------------------------------------------------------------


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


@kernel
def move_point(p: Point, out: np.ndarray):
    for i in range(out.shape[0]):
        p.x = p.x + 1.0
        out[i] = p.x


def test_a_structure_argument_reports_its_writes(gpu_backend):
    p = Point(1.0, 2.0)
    out = np.zeros(1, dtype=np.float32)
    move_point[gpu_backend](p, out)
    assert float(out[0]) == pytest.approx(2.0)
    assert p.x == pytest.approx(2.0), "the caller's own structure must receive the write"


# ---------------------------------------------------------------------------
# Writing an output and reading it back on the host in the same statement
# ---------------------------------------------------------------------------


@kernel
def square(a: np.ndarray):
    for i in range(a.shape[0]):
        a[i] = a[i] * a[i]


def test_read_your_own_write_on_the_host(gpu_backend):
    a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    square[gpu_backend](a)
    assert float(a[1]) == pytest.approx(9.0)
    # A second launch on the same array must see the new bytes, not a stale device copy.
    square[gpu_backend](a)
    assert float(a[1]) == pytest.approx(81.0)


# ---------------------------------------------------------------------------
# A view of an array (a slice) as the argument
# ---------------------------------------------------------------------------


def test_a_contiguous_slice_argument(gpu_backend):
    """`base[1:3]` is `[1.0, 2.0]`; squaring it must write those two slots of `base`."""
    base = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    view = base[1:3]
    square[gpu_backend](view)
    np.testing.assert_array_equal(base, np.array([0.0, 1.0, 4.0, 3.0], dtype=np.float32))


def test_a_strided_slice_argument(gpu_backend):
    """`base[::2]` is not contiguous; the values must still land in the right slots."""
    base = np.array([0.0, 100.0, 2.0, 300.0], dtype=np.float32)
    view = base[::2]
    square[gpu_backend](view)
    np.testing.assert_array_equal(base, np.array([0.0, 100.0, 4.0, 300.0], dtype=np.float32))
