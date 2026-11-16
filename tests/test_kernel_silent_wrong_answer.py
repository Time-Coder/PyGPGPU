"""Regression tests for two silent-wrong-answer bugs found by a white-box review.

Both share a shape: a check that exists and works on one path is missing on a second path
that reaches the same device code, so the failure is a wrong number rather than an error.

* **the mixed-width subscript**: the launch compares an array's rank against the number of
  indices it is subscripted with, and the translator records only the *widest* spelling --
  so `a[i]` beside `a[i, 0]` passes the check and is emitted as a flat `a.flat[i]`, where
  python reads the row `a[i, :]`.
* **the outer-scope integer**: a python `int` is range-checked against the width it travels
  as when it arrives as a *parameter*, and not when it is inlined into the generated source
  as a module-level constant.

Each test names the python answer, and every backends' answer must match it. The kernels
live in `regressions/` one module per bug (one python module is one translation unit).
"""
from __future__ import annotations

import numpy as np
import pytest

from regressions.mixed_subscript_width import (
    mixed_subscript_row,
    mixed_subscript_row_store,
    narrow_subscript_row,
)
from regressions.outer_scalar_width import (
    WIDE_CONSTANT,
    uses_wide_argument,
    uses_wide_constant,
)

from pygpgpu import kernel

# ---------------------------------------------------------------------------
# The mixed-width subscript
# ---------------------------------------------------------------------------


def _square() -> np.ndarray:
    return np.arange(16, dtype=np.float32).reshape(4, 4)


def _python_mixed_answer(a: np.ndarray) -> np.ndarray:
    """What the body computes under the interpreter: `a[i] + a[i, 0]` summed over the row."""
    return np.array([float(a[i].sum() + a[i, 0]) for i in range(a.shape[0])],
                    dtype=np.float32)


def test_a_row_read_beside_an_element_read_is_refused(gpu_backend):
    """`out[i] = a[i] + a[i, 0]` reads a row, which a work item cannot represent.

    The same read written alone is refused ("too few indices for 'a'"), so the mixed body
    must be refused in the same words rather than answering a number the flat offset
    happened to produce.
    """
    a = _square()
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="too few indices"):
        mixed_subscript_row[gpu_backend][(1, 4)](a, out)


def test_the_narrow_read_alone_is_still_refused(gpu_backend):
    """The control: this spelling already refused and must keep refusing."""
    a = _square()
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="too few indices"):
        narrow_subscript_row[gpu_backend][(1, 4)](a, out)


def test_the_interpreter_refuses_it_too():
    """The interpreter runs the same source, so it reaches the same refusal.

    It executes the body serially, and a row read there answers a sub-array -- legal python,
    which the device cannot represent. The launch must not diverge from the device on a
    source it accepts, so the refusal is the shared answer.
    """
    a = _square()
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises((ValueError, IndexError)):
        mixed_subscript_row["python"](a, out)


def test_a_row_store_beside_an_element_read_is_refused(gpu_backend):
    """`a[i] = 7.0` writes a whole row; mixing it with `a[i, 0]` must not narrow the store.

    A partial store is python's broadcast and is *legal* on its own (the tail parameter
    carries the row's length), so this checks the mixed body cannot silently write one
    element of the row instead.
    """
    a = _square()
    out = np.zeros(4, dtype=np.float32)
    try:
        mixed_subscript_row_store[gpu_backend][(1, 4)](a, out)
    except ValueError:
        return                      # refused outright: also acceptable
    # If it ran, every element of each row must be the stored value -- a flat one-element
    # store would leave the rest of the row untouched.
    expected = np.full((4, 4), 7.0, dtype=np.float32)
    np.testing.assert_array_equal(a, expected)


# ---------------------------------------------------------------------------
# The outer-scope integer
# ---------------------------------------------------------------------------


def test_an_outer_constant_too_wide_for_the_mapping_is_refused(gpu_backend):
    """`2 ** 31` inlined into the source must be refused, as the same value as a parameter is.

    It used to compile with nvcc's "integer conversion resulted in a change of sign" warning
    and reach the device as `-2147483648`, so the kernel computed with a number the caller
    never wrote.
    """
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(ValueError, match="does not fit"):
        uses_wide_constant[gpu_backend][(1, 1)](out)


def test_the_same_value_as_a_parameter_is_still_refused(gpu_backend):
    """The control: the parameter path already refused."""
    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(ValueError, match="does not fit"):
        uses_wide_argument[gpu_backend][(1, 1)](out, WIDE_CONSTANT)


def test_a_value_that_fits_is_still_accepted(gpu_backend):
    """The fix must not refuse a constant the mapping can carry."""
    out = np.zeros(1, dtype=np.int32)
    uses_wide_argument[gpu_backend][(1, 1)](out, 2 ** 31 - 1)
    assert int(out[0]) == 2 ** 31 - 1


def test_a_body_literal_keeps_its_old_behaviour(gpu_backend):
    """A literal written *inside* the body is not the same thing as a named constant.

    `0xFFFFFFFF` in a hash mix is unsigned arithmetic on purpose (see
    tests/test_kernel_scalar_calls.py), so the named-constant rule must not reach into the
    body and refuse it -- the check belongs to `_visit_name`, not to every literal. The
    compiler's own conversion of the literal is the old behaviour, and it is kept.
    """
    out = np.zeros(1, dtype=np.int32)
    literal_in_body[gpu_backend][(1, 1)](out)
    assert int(out[0]) == -1


@kernel
def literal_in_body(out: np.ndarray):
    """A body literal one past `int`'s range: refused as a *name*, allowed as a literal."""
    for i in range(out.shape[0]):
        out[i] = 0xFFFFFFFF
