# M2: the fused `grad` against `check_grad`, the two-call shape, and the refusals.
#
# `check_grad` runs the adjoint on the parametrized backend and the finite differences on
# the `python` backend, so a device disagreement is a translation bug and a disagreement
# on both is an AD bug.
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp
from pygpgpu import atomic_add, kernel
from pygpgpu.ad import check_grad


def _host(value: object) -> np.ndarray:
    """The plain numpy view of a gnp array, its device bytes pulled home."""
    if isinstance(value, gnp.ndarray):
        value._ensure_host_bytes()
        return value.view(np.ndarray)
    return np.asarray(value)


def test_single_store_body_grad(backend):
    """The saxpy shape: `grad` seeds the scalar return; `vjp` weights an array result."""
    @kernel
    def saxpy(a, x, y, out):
        for i in range(out.shape[0]):
            out[i] = a * x[i] + y[i]
        return out

    n = 16
    x = gnp.arange(n, dtype=np.float32)
    x.requires_grad = True
    y = gnp.arange(n, dtype=np.float32) * 0.5
    y.requires_grad = True
    out = gnp.zeros(n, np.float32)
    _result, pullback = saxpy.vjp()[backend](2.0, x, y, out)
    gradients = pullback(gnp.ones(n, np.float32))
    # d sum(out)/dx = a, d sum(out)/dy = 1
    np.testing.assert_allclose(_host(gradients[0]), 2.0, rtol=1e-6)
    np.testing.assert_allclose(_host(gradients[1]), 1.0, rtol=1e-6)


def test_two_statement_chain_grad(backend):
    """A chain through a local matches the analytic gradient."""
    @kernel
    def chain(x, out):
        for i in range(out.shape[0]):
            u = x[i] * 3.0
            out[i] = u * u + x[i]
        return out

    x = gnp.arange(1, 9, dtype=np.float32) * 0.5
    x.requires_grad = True
    out = gnp.zeros(8, np.float32)
    _result, pullback = chain.vjp()[backend](x, out)
    gradients = pullback(gnp.ones(8, np.float32))
    np.testing.assert_allclose(_host(gradients[0]),
                               18.0 * _host(x) + 1.0, rtol=1e-5)


def test_if_body_grad(backend):
    """The predicate is per work item and deterministic, so the adjoint re-evaluates it."""
    @kernel
    def hard_tanh(x, out):
        for i in range(out.shape[0]):
            if x[i] > 0.5:
                out[i] = x[i] * x[i]
            else:
                out[i] = 0.25
        return out

    x = (np.arange(8, dtype=np.float32) * 0.25)
    x_g = gnp.array(x, np.float32)
    x_g.requires_grad = True
    out = gnp.zeros(8, np.float32)
    result, pullback = hard_tanh.vjp()[backend](x_g, out)
    weights = np.ones(8, np.float32)
    gradients = pullback(gnp.array(weights, np.float32))
    expected = np.where(x > 0.5, 2.0 * x, 0.0).astype(np.float32)
    np.testing.assert_allclose(_host(gradients[0]), expected, rtol=1e-5)
    # the forward result is the caller's own object
    assert result is out


def test_while_body_grad(backend):
    """A `while` reverses by replay: the trip count is a private local, no memory."""
    @kernel
    def decay(x, out):
        for i in range(out.shape[0]):
            acc = 0.0
            v = x[i]
            n = 0
            while n < 4:
                acc += v
                v = v * 0.5
                n = n + 1
            out[i] = acc

    x = gnp.arange(8, dtype=np.float32) * 0.5
    x.requires_grad = True
    out = gnp.zeros(8, np.float32)
    result, pullback = decay.vjp(out="out")[backend](x, out)
    gradients = pullback(gnp.ones(8, np.float32))
    # forward: v + v/2 + v/4 + v/8 = 1.875 v; the derivative is the same constant
    np.testing.assert_allclose(_host(result), 1.875 * _host(x), rtol=1e-5)
    np.testing.assert_allclose(_host(gradients[0]), 1.875, rtol=1e-4)


def test_carried_scalar_return_needs_no_loss_named(backend):
    """A body that `return`s a carried scalar IS the loss; `wrt` is inferred."""
    @kernel
    def squared_error(x, target):
        total = 0.0
        for i in range(x.shape[0]):
            total += (x[i] - target[i]) ** 2
        return total

    x = gnp.arange(8, dtype=np.float32) * 0.25
    x.requires_grad = True
    target = gnp.zeros(8, np.float32)
    loss_value, gradients = squared_error.value_and_grad()[backend](x, target)
    np.testing.assert_allclose(_host(gradients[0]), 2.0 * _host(x), rtol=1e-5)
    np.testing.assert_allclose(float(loss_value), float(np.sum(_host(x) ** 2)),
                              rtol=1e-6)


def test_reduction_through_atomic_loss(backend):
    """`loss="name"` reads the number the body accumulated through atomic_add."""
    @kernel
    def sse(x, target, loss):
        for i in range(x.shape[0]):
            atomic_add(loss, 0, (x[i] - target[i]) ** 2)

    x = (np.arange(8, dtype=np.float32) * 0.1)
    x_g = gnp.array(x, np.float32)
    x_g.requires_grad = True
    target = gnp.zeros(8, np.float32)
    loss = gnp.zeros(1, np.float32)
    gradients = sse.grad(loss="loss")[backend](x_g, target, loss)
    np.testing.assert_allclose(_host(gradients[0]), 2.0 * x, rtol=1e-5)
    np.testing.assert_allclose(_host(loss), float(np.sum(x * x)), rtol=1e-5)


@pytest.mark.parametrize("backend_arg", ["python"])
def test_check_grad_passes_on_the_shapes(backend_arg):
    """The full check: AD on one backend against finite differences on the interpreter."""
    @kernel
    def branchy(x):
        total = 0.0
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                total += x[i] * x[i]
            else:
                total += 1.0
        return total

    x = gnp.array(np.arange(8, dtype=np.float32) - 2.5, np.float32)
    x.requires_grad = True
    # off the branch boundary, so the finite differences do not cross the break
    assert check_grad(branchy, x, backend=backend_arg) < 1e-3


def test_check_grad_passes_on_device_backends(gpu_backend):
    """The device adjoint against interpreter finite differences."""
    @kernel
    def squared_error(x, target):
        total = 0.0
        for i in range(x.shape[0]):
            total += (x[i] - target[i]) ** 2
        return total

    x = gnp.arange(8, dtype=np.float32) * 0.25
    x.requires_grad = True
    target = gnp.zeros(8, np.float32)
    assert check_grad(squared_error, x, target, backend=gpu_backend) < 1e-3


def test_grad_accumulates_into_grad_and_zero_grad_clears(backend):
    """`.grad` accumulates across calls (PyTorch semantics); `zero_grad()` resets."""
    @kernel
    def squared_error(x, target):
        total = 0.0
        for i in range(x.shape[0]):
            total += (x[i] - target[i]) ** 2
        return total

    x = gnp.arange(8, dtype=np.float32) * 0.25
    x.requires_grad = True
    target = gnp.zeros(8, np.float32)
    first = squared_error.grad()[backend](x, target)[0]
    second = squared_error.grad()[backend](x, target)[0]
    assert second is first
    # each call contributed 2 * (x - target); the buffer holds both
    np.testing.assert_allclose(_host(second), 4.0 * _host(x), rtol=1e-5)
    x.zero_grad()
    assert x.grad is None
    fresh = squared_error.grad()[backend](x, target)[0]
    np.testing.assert_allclose(_host(fresh), 2.0 * _host(x), rtol=1e-5)


def test_same_array_through_two_parameters_accumulates_both(backend):
    """`k(a, a)`: two gradient contributions per work item, both into one buffer."""
    @kernel
    def pair(a, out):
        for i in range(out.shape[0]):
            out[i] = a[i] * a[i]
        return out

    a = gnp.arange(4, dtype=np.float32) + 1.0
    a.requires_grad = True
    out = gnp.zeros(4, np.float32)
    _result, pullback = pair.vjp()[backend](a, out)
    gradients = pullback(gnp.ones(4, np.float32))
    # d sum(out)/da = 2a -- the two parameters' contributions land in one buffer
    np.testing.assert_allclose(_host(gradients[0]), 2.0 * _host(a), rtol=1e-5)
