"""Bug: a mixed-width subscript on one array bypasses the rank check and silently misreads.

`a[i]` on a 2-D array is python's *row* `a[i, :]`, which a work item cannot represent, so
the translator's own rule refuses it ("too few indices"). That refusal is driven by the
count of indices each array is subscripted with, and the count is recorded as the **widest**
spelling anywhere in the body (`targets._record_subscript_indices` keeps the maximum). A body
that also writes `a[i, 0]` therefore records 2, the launch's check sees `count == rank` and
lets the narrow read through -- and it is then emitted as the *flat* offset `a.flat[i]`.

The two spellings live in separate modules because one python module is one translation
unit: a compile-time refusal in one would mask the other.

``mixed_subscript_row`` is the failing case; ``narrow_subscript_row`` is the control that
already refused correctly, and it must keep refusing.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import get_global_id, kernel


@kernel
def mixed_subscript_row(a: np.ndarray, out: np.ndarray):
    """`a[i]` (a row) beside `a[i, 0]` (one element): the narrow read must still refuse."""
    i = get_global_id(0)
    out[i] = a[i] + a[i, 0]


@kernel
def narrow_subscript_row(a: np.ndarray, out: np.ndarray):
    """`a[i]` alone, which the launch already refuses with `too few indices`."""
    i = get_global_id(0)
    out[i] = a[i]


@kernel
def mixed_subscript_row_store(a: np.ndarray, out: np.ndarray):
    """The store side: `a[i] = ...` writes a whole row, and mixing it with `a[i, 0]` must
    not turn the row store into a single-element store."""
    i = get_global_id(0)
    a[i] = 7.0
    out[i] = a[i, 0]
