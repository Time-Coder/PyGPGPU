# M1: the fused `jvp` against hand-written derivatives, on every backend this machine has.
#
# The `python` backend executes the adjoint body as written, so a test that passes there
# and fails on a device is a translation bug; write every case over the conftest
# parametrization and read a failure accordingly.
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp
from pygpgpu import kernel, sqrt


def _host(value: object) -> np.ndarray:
    """The plain numpy view of a gnp array, its device bytes pulled home."""
    if isinstance(value, gnp.ndarray):
        value._ensure_host_bytes()
        return value.view(np.ndarray)
    return np.asarray(value)


def _jvp(kernel_object, x, dx, out, backend):
    """`k.jvp(tangent=dx)(x, out)` on the parametrized backend.

    `wrt` is inferred from the `requires_grad` flags, so the tracked array is flagged
    here the way a user flags it.
    """
    x.requires_grad = True
    form = kernel_object.jvp(tangent=gnp.array(dx, dtype=x.dtype), out="out")
    form = form[backend]
    return form(x, out)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_elementwise_polynomial_jvp(backend, dtype):
    """d/dt f(x + t*dx) at t=0 matches the hand-written derivative of each element."""
    @kernel
    def scale_shift(x, out):
        for i in range(out.shape[0]):
            u = x[i] * 3.0
            out[i] = u * u + x[i]

    values = (np.arange(1, 9) * 0.5).astype(dtype)
    x = gnp.array(values, dtype=dtype)
    out = gnp.zeros(8, dtype=dtype)
    tangent = _jvp(scale_shift, x, np.ones(8, dtype=dtype), out, backend)
    # d/dx (3x)^2 + x = 18x + 1, evaluated with dx = 1
    expected = (18.0 * values.astype(np.float64) + 1.0).astype(dtype)
    np.testing.assert_allclose(_host(tangent), expected,
                               rtol=1e-5 if dtype != np.float64 else 1e-12)


def test_jvp_direction_scales_the_tangent(backend):
    """The tangent answers `d out / d t` along `dx`, so dx = 2 doubles it."""
    @kernel
    def twice(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    x = gnp.arange(4, dtype=np.float32)
    out = gnp.zeros(4, np.float32)
    ones = _jvp(twice, x, np.ones(4, np.float32), out, backend)
    twos = _jvp(twice, x, np.full(4, 2.0, np.float32), out, backend)
    np.testing.assert_allclose(_host(ones), 2.0, rtol=1e-6)
    np.testing.assert_allclose(_host(twos), 4.0, rtol=1e-6)


def test_jvp_math_builtin_chain(backend):
    """A chain through a ruled math builtin matches the analytic derivative."""
    @kernel
    def hypot_like(x, out):
        for i in range(out.shape[0]):
            out[i] = sqrt(x[i] * x[i] + 1.0)

    values = np.array([0.5, 1.0, 2.0, 4.0], np.float32)
    x = gnp.array(values, np.float32)
    out = gnp.zeros(4, np.float32)
    tangent = _jvp(hypot_like, x, np.ones(4, np.float32), out, backend)
    # d/dx sqrt(x^2 + 1) = x / sqrt(x^2 + 1)
    expected = values / np.sqrt(values * values + 1.0)
    np.testing.assert_allclose(_host(tangent), expected, rtol=1e-5)


def test_jvp_scalar_return(backend):
    """A reduction's tangent comes home through the return slot."""
    @kernel
    def squared_sum(x):
        total = 0.0
        for i in range(x.shape[0]):
            total += x[i] * x[i]
        return total

    x = gnp.arange(8, dtype=np.float32)
    x.requires_grad = True
    form = squared_sum.jvp(tangent=np.ones(8, np.float32))[backend]
    value = form(x)
    # d/dt sum((x + t)^2) at t=0 = sum(2x)
    np.testing.assert_allclose(float(value), float(np.sum(2.0 * np.arange(8))),
                               rtol=1e-5)


def test_jvp_refuses_in_place_write_to_tracked():
    """A tracked array is read-only in a differentiable body, in either mode."""
    @kernel
    def update(x, out):
        for i in range(out.shape[0]):
            x[i] = x[i] * 2.0
            out[i] = x[i]

    x = gnp.ones(4, np.float32)
    x.requires_grad = True
    out = gnp.zeros(4, np.float32)
    form = update.jvp(tangent=np.ones(4, np.float32))["python"]
    with pytest.raises(SyntaxError):
        form(x, out)


def test_jvp_positional_configuring_call_refused():
    """`k.jvp(x, out)` is a positional configuring call: refused by name."""
    @kernel
    def twice(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    x = gnp.ones(4, np.float32)
    out = gnp.zeros(4, np.float32)
    with pytest.raises(SyntaxError):
        twice.jvp(x, out)

def test_branch_jvp_matches_the_taken_branch(backend):
    """A branch in the dispatch: the tangent follows the branch the forward takes.

    Regression: `transform_if` passed two arguments to a one-argument
    `transform_block`, so any `if` in the dispatch crashed with a raw TypeError
    instead of translating.
    """
    @kernel
    def branchy(x, out):
        for i in range(out.shape[0]):
            u = x[i]
            if u > 0.0:
                out[i] = u * u
            else:
                out[i] = -u

    values = np.array([2.0, -3.0, 0.5, -1.0], dtype=np.float32)
    x = gnp.array(values, dtype=np.float32)
    out = gnp.zeros(4, dtype=np.float32)
    tangent = _jvp(branchy, x, np.ones(4, dtype=np.float32), out, backend)
    # d/dx (x^2) = 2x where x > 0; d/dx (-x) = -1 where x <= 0
    expected = np.where(values > 0.0, 2.0 * values, -1.0).astype(np.float32)
    np.testing.assert_allclose(_host(tangent), expected, rtol=1e-5)
