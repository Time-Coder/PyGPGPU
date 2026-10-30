"""The `gpu.atomic_*` family: what it covers, and what it refuses.

Three things are pinned here:

* `atomic_cas` exists and is portable (`atomicCAS` on CUDA, `atomic_cmpxchg` on
  OpenCL), returning the previous value like the other atomics.
* 64-bit element types work on every backend. CUDA has no 64-bit `atomicSub` and no
  signed 64-bit `atomicAdd`, so the translator writes those through
  `unsigned long long`, which is bit-identical for add/sub/bitwise/exchange/CAS.
* A rejected element type is named, instead of claiming "only integer arrays" while
  refusing `uchar`, which is an integer.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel, atomic_add, atomic_cas, atomic_max, atomic_min, atomic_sub, atomic_xchg, atomic_xor


# ---------------------------------------------------------------------------
# atomic_cas
# ---------------------------------------------------------------------------


@kernel
def cas_claim_once(out: np.ndarray):
    """Classic claim: the first work item swaps 0 -> 1, every later one sees 1."""
    for i in range(8):
        atomic_cas(out, 0, 0, 1)


@kernel
def cas_returns_previous(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        out[1] = atomic_cas(out, 0, x[i], x[i] + 100)


def test_cas_claims_a_slot_exactly_once(runner):
    out = np.zeros(1, dtype=np.int32)
    runner(cas_claim_once, out)
    np.testing.assert_array_equal(out, np.array([1], dtype=np.int32))


def test_cas_returns_the_previous_value(runner):
    """One work item: the slot is 0, so the compare succeeds and 5 is reported as old."""
    x = np.array([0], dtype=np.int32)
    out = np.array([0, -1], dtype=np.int32)
    runner(cas_returns_previous, x, out)
    np.testing.assert_array_equal(out, np.array([100, 0], dtype=np.int32))


def test_cas_leaves_the_value_when_the_compare_fails(runner):
    x = np.array([7], dtype=np.int32)
    out = np.array([0, -1], dtype=np.int32)
    runner(cas_returns_previous, x, out)
    np.testing.assert_array_equal(out, np.array([0, 0], dtype=np.int32))


# ---------------------------------------------------------------------------
# 64-bit element types
# ---------------------------------------------------------------------------


@kernel
def add_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_add(out, 0, x[i])


@kernel
def sub_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_sub(out, 0, x[i])


@kernel
def min_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_min(out, 0, x[i])


@kernel
def max_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_max(out, 0, x[i])


@kernel
def xchg_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_xchg(out, 0, x[i])


@kernel
def xor_wide(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_xor(out, 0, x[i])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_sixty_four_bit_add(runner, dtype):
    x = np.array([1 << 40, 1, 2], dtype=dtype)
    out = np.zeros(1, dtype=dtype)
    runner(add_wide, x, out)
    assert int(out[0]) == (1 << 40) + 3


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_sixty_four_bit_sub(runner, dtype):
    x = np.array([1, 2, 3], dtype=dtype)
    out = np.full(1, 10, dtype=dtype)
    runner(sub_wide, x, out)
    assert int(out[0]) == 4


def test_sixty_four_bit_sub_of_a_negative_value(runner):
    """10 - (-5) = 15, which the unsigned-negation route must still get right."""
    x = np.array([-5], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    runner(sub_wide, x, out)
    assert int(out[0]) == 5


def test_sixty_four_bit_min_uses_the_signed_order(runner):
    """Unsigned ordering would pick 3, not -5, so this catches a plain unsigned cast."""
    x = np.array([-5, 3], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    runner(min_wide, x, out)
    assert int(out[0]) == -5


def test_sixty_four_bit_max_keeps_wide_values(runner):
    x = np.array([5, 1 << 40], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    runner(max_wide, x, out)
    assert int(out[0]) == 1 << 40


def test_sixty_four_bit_unsigned_min_keeps_wide_values(runner):
    """Above 2^32 the value must not be truncated, which is what a `long` type did."""
    x = np.array([5, 1 << 40], dtype=np.uint64)
    out = np.full(1, 1 << 50, dtype=np.uint64)
    runner(min_wide, x, out)
    assert int(out[0]) == 5


def test_sixty_four_bit_exchange_and_xor(runner):
    x = np.array([7, 8], dtype=np.uint64)
    out = np.zeros(1, dtype=np.uint64)
    runner(xchg_wide, x, out)
    assert int(out[0]) == 8

    x = np.array([0b1100, 0b1010], dtype=np.uint64)
    out = np.zeros(1, dtype=np.uint64)
    runner(xor_wide, x, out)
    assert int(out[0]) == 0b0110


# ---------------------------------------------------------------------------
# int64 arrays in general: the same 64-bit widening applies to ordinary code
# ---------------------------------------------------------------------------


@kernel
def copy_int64(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = x[i] * 2


def test_int64_arrays_are_wide_on_every_backend(runner):
    """The element stride is 8 bytes; a `long` parameter would have made it 4."""
    x = np.array([1 << 40, -(1 << 41)], dtype=np.int64)
    out = np.zeros(2, dtype=np.int64)
    runner(copy_int64, x, out)
    np.testing.assert_array_equal(out, x * 2)


# ---------------------------------------------------------------------------
# A refused element type is named
# ---------------------------------------------------------------------------


@kernel
def add_byte(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_add(out, 0, x[i])


def test_a_byte_sized_element_is_refused_with_a_useful_message(gpu_backend, runner):
    with pytest.raises(SyntaxError) as caught:
        runner(add_byte, np.array([1, 2], dtype=np.uint8), np.zeros(1, dtype=np.uint8))
    message: str = str(caught.value)
    assert "int32" in message and "int64" in message
    assert "uchar" in message


def test_the_message_no_longer_claims_only_integers_are_allowed(gpu_backend, runner):
    """`uchar` *is* an integer; the old wording was misleading."""
    with pytest.raises(SyntaxError) as caught:
        runner(add_byte, np.array([1], dtype=np.uint8), np.zeros(1, dtype=np.uint8))
    assert "only supports integer arrays" not in str(caught.value)
