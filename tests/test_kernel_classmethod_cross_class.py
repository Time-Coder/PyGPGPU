"""Cross-class classmethod calls, and the receiver passed through a helper.

The two shapes the main classmethod suite does not cover: a kernel (or a helper
it called) reaching a classmethod of a class it does not hold -- the callee's
cls binds to that class, whose scalar attributes travel as compile-time
constants -- and a kernel handing its own receiver to a helper as a value,
whose r.doubled(v) resolves the member reads from the same class.
"""
from __future__ import annotations

import numpy as np

from pygpgpu import kernel


class _ForeignUtils:
    scale = 2.0
    offset = 10.0

    @classmethod
    def cm(cls, v):
        return v * cls.scale + cls.offset

    @classmethod
    def cm2(cls, v):
        # A nested call on the same foreign cls: the constants answer it too.
        return cls.cm(v) + 1.0


@kernel
def _cross_class_call(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = _ForeignUtils.cm(x[i])


@kernel
def _cross_class_nested(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = _ForeignUtils.cm2(x[i])


def _helper_cm(v):
    return _ForeignUtils.cm(v) + 0.5


@kernel
def _cross_class_via_helper(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = _helper_cm(x[i])


class _ReceiverThroughHelper:
    gain = 2.0

    @kernel
    def run(self, x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = dbl_helper(self, x[i])

    def doubled(self, v):
        return v * self.gain


def dbl_helper(r, v):
    return r.doubled(v)


def test_a_cross_class_classmethod_call_reads_the_foreign_constants(runner):
    """`BUtils.cm(x)` from a kernel body: the callee's cls is the foreign class,
    whose scalar attributes travel as compile-time constants -- no receiver
    struct exists for a class the caller does not hold."""
    out = np.zeros(3, dtype=np.float32)
    runner(_cross_class_call, np.ones(3, dtype=np.float32), out)
    np.testing.assert_allclose(out, 12.0)


def test_a_nested_cross_class_call_reads_them_too(runner):
    """`BUtils.cm2(v)` calls `cls.cm(v)` on the same foreign cls; the nested
    callee's constant reads answer the same table."""
    out = np.zeros(3, dtype=np.float32)
    runner(_cross_class_nested, np.ones(3, dtype=np.float32), out)
    np.testing.assert_allclose(out, 13.0)


def test_a_helper_calls_a_cross_class_classmethod(runner):
    """The same call from inside a helper the kernel called: the constants ride
    through the helper's own translation."""
    out = np.zeros(3, dtype=np.float32)
    runner(_cross_class_via_helper, np.ones(3, dtype=np.float32), out)
    np.testing.assert_allclose(out, 12.5)


def test_the_receiver_passes_through_a_helper(runner):
    """`dbl(self, x[i])` hands the receiver itself to a helper; the helper's
    `r.doubled(v)` resolves the member reads from the same class."""
    instance = _ReceiverThroughHelper()
    out = np.zeros(3, dtype=np.float32)
    runner(instance.run, np.ones(3, dtype=np.float32), out)
    np.testing.assert_allclose(out, 2.0)
