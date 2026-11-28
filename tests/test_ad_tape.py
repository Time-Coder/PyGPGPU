# M4: `Tape` across kernels -- launch recording, seeding, epoch validation, adjoints in
# reverse order, scatter adjoints through atomics.
#
# The tape records device launches only (the interpreter answers from the first launch
# phase and never reaches the hook), so every test takes `gpu_backend`.
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp
from pygpgpu import Tape, kernel


def _host(value: object) -> np.ndarray:
    """The plain numpy view of a gnp array, its device bytes pulled home."""
    if isinstance(value, gnp.ndarray):
        value._ensure_host_bytes()
        return value.view(np.ndarray)
    return np.asarray(value)


def test_two_kernel_chain_matches_the_reference(gpu_backend):
    """Forward + loss: the tape's gradients match the analytic composition."""
    @kernel
    def saxpy(a, x, y, out):
        for i in range(out.shape[0]):
            out[i] = a * x[i] + y[i]

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    n = 16
    x = gnp.arange(n, dtype=np.float32)
    x.requires_grad = True
    y = gnp.arange(n, dtype=np.float32) * 0.5
    y.requires_grad = True
    out = gnp.zeros(n, np.float32)
    loss = gnp.zeros(1, np.float32)

    with Tape() as tape:
        saxpy[gpu_backend](2.0, x, y, out)
        dot_loss[gpu_backend](out, loss)

    tape.backward(loss)
    # loss = sum((2x + y)^2); d/dx = 2 * out * 2, d/dy = 2 * out
    out_values = _host(out)
    np.testing.assert_allclose(_host(x.grad), 4.0 * out_values, rtol=1e-4)
    np.testing.assert_allclose(_host(y.grad), 2.0 * out_values, rtol=1e-4)


def test_mutated_after_forward_is_refused_naming_the_array(gpu_backend):
    """A caller overwrite between the passes is a wrong number with no crash: refused."""
    @kernel
    def saxpy(a, x, y, out):
        for i in range(out.shape[0]):
            out[i] = a * x[i] + y[i]

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    x = gnp.arange(4, dtype=np.float32)
    x.requires_grad = True
    y = gnp.zeros(4, np.float32)
    out = gnp.zeros(4, np.float32)
    loss = gnp.zeros(1, np.float32)

    with Tape() as tape:
        saxpy[gpu_backend](2.0, x, y, out)
        dot_loss[gpu_backend](out, loss)

    x *= 2.0        # the host bytes no longer describe what the forward used
    with pytest.raises(RuntimeError, match="epoch"):
        tape.backward(loss)


def test_reduction_adjoint_adds_once_per_element(gpu_backend):
    """The loss kernel's own accumulation is skipped; the cotangent flows once."""
    @kernel
    def fill(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    x = gnp.arange(4, dtype=np.float32) + 1.0
    x.requires_grad = True
    out = gnp.zeros(4, np.float32)
    loss = gnp.zeros(1, np.float32)

    with Tape() as tape:
        fill[gpu_backend](x, out)
        dot_loss[gpu_backend](out, loss)

    tape.backward(loss)
    # loss = sum((2x)^2) = 4 sum(x^2); d/dx = 8x
    np.testing.assert_allclose(_host(x.grad), 8.0 * _host(x), rtol=1e-4)
    # the forward's accumulation was skipped: the loss value itself is not doubled
    np.testing.assert_allclose(_host(loss), float(np.sum((2.0 * _host(x)) ** 2)),
                               rtol=1e-4)


def test_same_array_through_two_parameters_accumulates(gpu_backend):
    """`k(a, a)` is bound once by the launch but carries two gradient contributions."""
    @kernel
    def saxpy(a, x, y, out):
        for i in range(out.shape[0]):
            out[i] = a * x[i] + y[i]

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    a = gnp.arange(4, dtype=np.float32) + 1.0
    a.requires_grad = True
    out = gnp.zeros(4, np.float32)
    loss = gnp.zeros(1, np.float32)

    with Tape() as tape:
        saxpy[gpu_backend](2.0, a, a, out)
        dot_loss[gpu_backend](out, loss)

    tape.backward(loss)
    # d/d a[(2a + a)^2 sum] = 2 * out * 3 = 6 * out
    np.testing.assert_allclose(_host(a.grad), 6.0 * _host(out), rtol=1e-4)


def test_tape_reset_clears_cotangents(gpu_backend):
    """`reset()` drops the cotangent buffers; a second backward starts clean."""
    @kernel
    def fill(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    x = gnp.arange(4, dtype=np.float32) + 1.0
    x.requires_grad = True
    out = gnp.zeros(4, np.float32)
    loss = gnp.zeros(1, np.float32)

    with Tape() as tape:
        fill[gpu_backend](x, out)
        dot_loss[gpu_backend](out, loss)

    tape.backward(loss)
    first = _host(x.grad).copy()
    tape.reset()
    x.zero_grad()
    tape.backward(loss)
    np.testing.assert_allclose(_host(x.grad), first, rtol=1e-5)


def test_tapes_do_not_nest():
    """One tape owns the launches of its region; nesting is refused, not shadowed."""
    with Tape():
        with pytest.raises(RuntimeError, match="already active"):
            with Tape():
                pass

def test_method_kernel_launch_records_the_receiver(gpu_backend):
    """A tape over a method kernel: the receiver rides with the record, and the
    backward launch -- whose adjoint keeps the origin's signature, receiver included
    -- gets it prepended.

    Regression: the receiver was never recorded, so the adjoint launch was short one
    argument and crashed with a raw TypeError (or, with another signature shape,
    misbound silently).
    """

    class Accumulator:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        @kernel
        def accumulate(self, x, out):
            for i in range(out.shape[0]):
                out[i] = self.scale * x[i]

    n = 8
    x = gnp.arange(n, dtype=np.float32)
    x.requires_grad = True
    out = gnp.zeros(n, np.float32)
    acc = Accumulator(3.0)

    with Tape() as tape:
        acc.accumulate[gpu_backend](x, out)

    tape.backward(out)
    # loss = sum(3 * x): d/dx = 3, per element
    np.testing.assert_allclose(_host(x.grad), 3.0, rtol=1e-5)
