"""Accumulating into a single shared scalar from a `@kernel` body.

A handwritten kernel takes a pointer to one element and passes it to `atomicAdd`, e.g.
``__global int* total``. The same three shapes work here, and this module pins all of them:

* a **1-element array** argument is that pointer (``atomic_add(total, 0, 1)``);
* a **`nonlocal`/`global` scalar** is turned into a device ``T*`` by the outer-scope
  machinery, so ``counter += 1`` is one shared location that every work item updates;
* a **counter created in the body** (``np.zeros(1)``) can be returned as the result.

The second one used to be silently wrong: the store was emitted as ``(*counter) += 1``, so
the static read/write analysis did not see a write to the parameter, the buffer was never
read back, and the host kept its initial 0. Even once written back, a plain
read-modify-write loses increments when thousands of work items share the location. Both
are fixed: the store is a subscript (visible to the analysis) and a shared scalar is always
emitted as an atomic operation, because it can never be private to one work item.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import atomic_add, kernel

# Module globals the `global` tests write; each test resets them.
TOTAL: int = 0
SIGNED_TOTAL: int = 0
BITS: int = 0


# ---------------------------------------------------------------------------
# A 1-element array is the pointer a handwritten kernel would take
# ---------------------------------------------------------------------------


@kernel
def count_into_argument(x: np.ndarray, total: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            atomic_add(total, 0, 1)


def test_a_one_element_argument_is_a_shared_counter(runner):
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    total = np.zeros(1, dtype=np.int32)
    runner(count_into_argument, x, total)
    assert int(total[0]) == int((x > 0.0).sum())


def test_the_plus_equals_spelling_is_atomic_too(runner):
    """`total[0] += 1` has a constant index, so it is not provably private: it is atomic."""
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    total = np.zeros(1, dtype=np.int32)
    runner(count_into_argument_with_plus_equals, x, total)
    assert int(total[0]) == int((x > 0.0).sum())


@kernel
def count_into_argument_with_plus_equals(x: np.ndarray, total: np.ndarray):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            total[0] += 1


# ---------------------------------------------------------------------------
# A `global`/`nonlocal` scalar: one location shared by every work item
# ---------------------------------------------------------------------------


@kernel
def count_with_global(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            TOTAL += 1


@kernel
def subtract_with_global(x: np.ndarray):
    global SIGNED_TOTAL
    for i in range(x.shape[0]):
        if x[i] < 0.0:
            SIGNED_TOTAL -= 1


@kernel
def or_with_global(x: np.ndarray):
    global BITS
    for i in range(x.shape[0]):
        BITS |= int(x[i])


def _reset_globals() -> None:
    global TOTAL, SIGNED_TOTAL, BITS
    TOTAL = 0
    SIGNED_TOTAL = 0
    BITS = 0


def test_an_outer_scalar_increment_is_exact(runner):
    """4096 work items all incrementing one location: not one may be lost."""
    _reset_globals()
    x = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    runner(count_with_global, x)
    assert TOTAL == int((x > 0.0).sum())


def test_an_outer_scalar_is_written_back(runner):
    """The host value has to change: the increment lives in a device buffer until then."""
    _reset_globals()
    runner(count_with_global, np.ones(8, dtype=np.float32))
    assert TOTAL == 8


def test_an_outer_scalar_decrement_works(runner):
    _reset_globals()
    runner(subtract_with_global, np.array([-1.0, -2.0, 3.0], dtype=np.float32))
    assert SIGNED_TOTAL == -2


def test_a_bitwise_outer_scalar_update_works(runner):
    _reset_globals()
    runner(or_with_global, np.array([1, 2, 4], dtype=np.int32))
    assert BITS == 7


def test_a_nonlocal_scalar_works_the_same_way(runner):
    """Closure cells take the same route as module globals."""
    total = 0

    @kernel
    def count_with_nonlocal(x: np.ndarray):
        nonlocal total
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                total += 1

    x = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    runner(count_with_nonlocal, x)
    assert total == int((x > 0.0).sum())


# ---------------------------------------------------------------------------
# A counter created in the body and returned
# ---------------------------------------------------------------------------


@kernel
def count_and_return(x: np.ndarray):
    """No argument needed: the body creates the counter, fills it, and hands it back."""
    total: np.ndarray = np.zeros(1, dtype=np.int32)
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            total[0] += 1
    return total


def test_a_created_counter_is_returned(runner):
    x = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    result = runner(count_and_return, x)
    assert result.shape == (1,)
    assert int(result[0]) == int((x > 0.0).sum())


# ---------------------------------------------------------------------------
# The wrong tool is refused with advice
# ---------------------------------------------------------------------------


@kernel
def misuse_atomic_on_scalar(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        atomic_add(TOTAL, 0, 1)


def test_atomic_add_on_an_outer_scalar_explains_the_right_spelling(gpu_backend, runner):
    _reset_globals()
    with pytest.raises(SyntaxError) as caught:
        runner(misuse_atomic_on_scalar, np.ones(4, dtype=np.float32))
    message: str = str(caught.value)
    assert "shared value" in message
    assert "TOTAL += value" in message


@kernel
def multiply_outer_scalar(x: np.ndarray):
    global TOTAL
    for i in range(x.shape[0]):
        TOTAL *= int(x[i])


def test_an_unsupported_shared_update_is_refused(gpu_backend, runner):
    """There is no atomic multiply, so it cannot be made exact."""
    _reset_globals()
    with pytest.raises(SyntaxError) as caught:
        runner(multiply_outer_scalar, np.ones(4, dtype=np.int32))
    assert "*=" in str(caught.value)
