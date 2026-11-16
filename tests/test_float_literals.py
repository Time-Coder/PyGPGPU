"""Bugs in float literal emission: infinities, NaN, and a value a float32 cannot hold.

The translator writes a python float literal into the C source. Three values break that:

* `float("inf")` was emitted as the bare word ``inf``, which is not a C literal at all --
  the compiler reports an undeclared identifier, so a kernel that used an infinity did not
  build;
* `float("nan")` became ``nanf``, a *call* that needs ``<math.h>`` -- present on one backend's
  prelude and not on another's;
* a value below the float32 range (``1e-40``) was spelled ``1e-40f``, which the **compiler**
  rounds to zero, so a device-only backend computed 0.0 where python computes 1e-40. The
  interpreter, which never converts the literal, kept python's number.

The fix is one function for both widths; these tests pin what it must answer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pygpgpu import kernel


def _literal_code(value: float, width: str) -> str:
    """The C text the translator emits for one real literal, straight from the helper."""
    from pygpgpu.kernel.translator.expressions import _ExpressionsMixin

    class _Probe(_ExpressionsMixin):
        """An empty translator: `_real_literal` needs no other state."""

    return _Probe()._real_literal(value, width)


# ---------------------------------------------------------------------------
# The emitted text itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", ["float", "double"])
def test_an_infinity_is_emitted_as_a_c_literal(width):
    """`inf` is not a C token; the source needs a spelling the compiler accepts."""
    for value in (math.inf, -math.inf):
        text = _literal_code(value, width)
        assert "inf" not in text.replace("INFINITY", ""), f"emitted {text!r}"
        assert "INFINITY" in text or "1.0" in text or "0.0" in text, f"emitted {text!r}"


@pytest.mark.parametrize("width", ["float", "double"])
def test_a_nan_is_emitted_without_a_function_call(width):
    """`nanf` needs `<math.h>`; a self-contained expression does not."""
    text = _literal_code(math.nan, width)
    assert "nanf" not in text.lower(), f"emitted {text!r} (a call needs math.h)"
    assert "NAN" in text or "0.0" in text, f"emitted {text!r}"


def test_a_normal_float_keeps_its_suffix():
    """The control: ordinary literals are unchanged, so the fix is narrow."""
    assert _literal_code(0.5, "float") == "0.5f"
    assert _literal_code(0.5, "double") == "0.5"


# ---------------------------------------------------------------------------
# End to end: the device must answer python's number
# ---------------------------------------------------------------------------


@kernel
def inf_literal(out: np.ndarray):
    """`1e400` is a python *literal* that overflows to an infinity."""
    for i in range(out.shape[0]):
        out[i] = 1e400


@kernel
def neg_inf_literal(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = -1e400


@kernel
def tiny_literal(out: np.ndarray):
    """`1e-40` is below float32's range: the interpreter keeps it, a `1e-40f` literal is 0."""
    for i in range(out.shape[0]):
        out[i] = 1e-40


def test_an_infinity_literal_reaches_the_device(gpu_backend):
    out = np.zeros(1, dtype=np.float32)
    inf_literal[gpu_backend](out)
    assert math.isinf(float(out[0])) and out[0] > 0


def test_a_negative_infinity_literal_reaches_the_device(gpu_backend):
    out = np.zeros(1, dtype=np.float32)
    neg_inf_literal[gpu_backend](out)
    assert math.isinf(float(out[0])) and out[0] < 0


def test_a_subnormal_float64_literal_is_not_silently_zeroed(gpu_backend):
    """The literal is a float64 in python; writing it into float32 may flush, reading it
    back as a double must not turn it into zero on a float64 target."""
    out = np.zeros(1, dtype=np.float64)
    tiny_literal[gpu_backend](out)
    assert float(out[0]) != 0.0, "a float64 target must keep python's 1e-40"
