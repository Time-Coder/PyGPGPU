# M5: where the data lives -- receiver members, object members, outer-scope values,
# class constants -- and the deferred launch spellings.
#
# The rule every test here pins: a value the fused adjoint cannot carry a gradient slot
# for is REFUSED with the reason and the alternative spelling, never silently zero.
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


def _tracked(n:int = 8)->gnp.ndarray:
    x = gnp.arange(n, dtype=np.float32)
    x.requires_grad = True
    return x


def test_tracked_receiver_member_refused_at_the_call():
    """`self.points.requires_grad` would be silently ignored by wrt inference: refused."""
    out = gnp.zeros(8, np.float32)

    class Tracer:
        def __init__(self)->None:
            self.points = _tracked()

        @kernel
        def trace(self, out):
            for i in range(out.shape[0]):
                out[i] = self.points[i] * 2.0
            return out

    tracer = Tracer()
    with pytest.raises(SyntaxError, match="members are not differentiable"):
        tracer.trace.vjp()(out)


def test_tracked_object_member_refused_at_the_call():
    """The same refusal for a structure argument's ndarray member."""
    out = gnp.zeros(8, np.float32)

    class Bag:
        def __init__(self)->None:
            self.field = _tracked()

    @kernel
    def body(bag, out):
        for i in range(out.shape[0]):
            out[i] = bag.field[i] * 2.0
        return out

    with pytest.raises(SyntaxError, match="pass the member as its own array argument"):
        body.vjp()(Bag(), out)


def test_member_store_refused():
    """A store into `self.out[i]` is invisible to a naive write analysis: refused.

    The loss is the carried scalar, so the form builds, and the scan-level guard is what
    names the member store -- the case a naive write analysis would silently drop.
    """
    x = _tracked()

    class Holder:
        def __init__(self)->None:
            self.out = gnp.zeros(8, np.float32)

        @kernel
        def fill(self, x):
            total = 0.0
            for i in range(x.shape[0]):
                self.out[i] = x[i] * 2.0
                total += x[i]
            return total

    holder = Holder()
    with pytest.raises(SyntaxError, match="member"):
        holder.fill.grad()(x)


def test_unflagged_member_scalar_read_is_a_constant():
    """A member scalar nobody flagged is a constant to the adjoint: the tracked gradient
    carries its factor and the member's own (unrequested) gradient is dropped."""
    x = _tracked()
    out = gnp.zeros(8, np.float32)

    class Config:
        def __init__(self)->None:
            self.scale = 2.0

    @kernel
    def body(x, config, out):
        for i in range(out.shape[0]):
            out[i] = config.scale * x[i]
        return out

    _result, pullback = body.vjp()(x, Config(), out)
    gradients = pullback(gnp.ones(8, np.float32))
    np.testing.assert_allclose(_host(gradients[0]), 2.0, rtol=1e-6)


def test_outer_array_refused_naming_the_parameter_spelling():
    """An enclosing-scope array is refused with the pass-it-as-a-parameter message."""
    x = _tracked()
    out = gnp.zeros(8, np.float32)
    outer = gnp.ones(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = outer[i] * x[i]
        return out

    _result, pullback = body.vjp()(x, out)
    with pytest.raises(SyntaxError, match="enclosing scope"):
        pullback(gnp.ones(8, np.float32))


def test_outer_scalar_contributes_no_dead_slot():
    """An outer scalar's gradient was not requested: dropped, and the tracked gradient
    stays correct."""
    x = _tracked()
    out = gnp.zeros(8, np.float32)
    scale = 2.0        # an enclosing-scope scalar

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = scale * x[i]
        return out

    _result, pullback = body.vjp()(x, out)
    gradients = pullback(gnp.ones(8, np.float32))
    np.testing.assert_allclose(_host(gradients[0]), 2.0, rtol=1e-6)


def test_multi_pass_body_differentiates_through_the_intermediate():
    """Two passes chained by a caller-passed intermediate: forward in order, adjoints in
    reverse, the intermediate's cotangent buffer connecting them."""
    @kernel
    def two_pass(x, scratch, out):
        for i in range(scratch.shape[0]):
            scratch[i] = x[i] * 2.0
        for i in range(out.shape[0]):
            out[i] = scratch[i] * scratch[i]
        return out

    x = _tracked()
    scratch = gnp.zeros(8, np.float32)
    out = gnp.zeros(8, np.float32)
    result, pullback = two_pass.vjp()(x, scratch, out)
    np.testing.assert_allclose(_host(result), 4.0 * _host(x) ** 2, rtol=1e-5)
    gradients = pullback(gnp.ones(8, np.float32))
    # d sum(out)/dx = 2 * scratch * 2 = 8x
    np.testing.assert_allclose(_host(gradients[0]), 8.0 * _host(x), rtol=1e-5)


def test_multi_pass_grad_through_loss_array():
    """`loss="name"` on a two-pass body: the loss accumulation lives in the last pass."""
    @kernel
    def two_pass_sse(x, scratch, loss):
        for i in range(scratch.shape[0]):
            scratch[i] = x[i] * 2.0
        for i in range(scratch.shape[0]):
            loss[0] += scratch[i] * scratch[i]

    x = _tracked()
    scratch = gnp.zeros(8, np.float32)
    loss = gnp.zeros(1, np.float32)
    gradients = two_pass_sse.grad(loss="loss")(x, scratch, loss)
    # loss = sum((2x)^2) = 4 sum(x^2); d/dx = 8x
    np.testing.assert_allclose(_host(gradients[0]), 8.0 * _host(x), rtol=1e-5)
    np.testing.assert_allclose(_host(loss), float(np.sum((2.0 * _host(x)) ** 2)),
                               rtol=1e-4)


def test_multi_pass_refusals():
    """The shapes the multi-pass form cannot carry, each with its reason."""
    x = _tracked()
    scratch = gnp.zeros(8, np.float32)
    out = gnp.zeros(8, np.float32)

    @kernel
    def with_gap(x, scratch, out):
        for i in range(scratch.shape[0]):
            scratch[i] = x[i] * 2.0
        mid = scratch[0] * 1.0
        for i in range(out.shape[0]):
            out[i] = scratch[i] * scratch[i] + mid
        return out

    with pytest.raises(SyntaxError, match="between its magic loops"):
        with_gap.vjp()(x, scratch, out)[1](gnp.ones(8, np.float32))

    @kernel
    def created_intermediate(x, out):
        scratch = gnp.zeros(8, np.float32)
        for i in range(scratch.shape[0]):
            scratch[i] = x[i] * 2.0
        for i in range(out.shape[0]):
            out[i] = scratch[i] * scratch[i]
        return out

    # The phase machinery refuses the FORWARD itself first (a phase kernel does not
    # receive a prefix-created array), so the AD guard below it is a backstop for shapes
    # whose forward compiles; either way the shape never differentiates silently.
    with pytest.raises(SyntaxError):
        created_intermediate.vjp()(x, out)[1](gnp.ones(8, np.float32))

    @kernel
    def carried_loss(x, scratch):
        total = 0.0
        for i in range(scratch.shape[0]):
            scratch[i] = x[i] * 2.0
            total += scratch[i]
        for i in range(scratch.shape[0]):
            total += scratch[i]
        return total

    with pytest.raises(SyntaxError, match="carried-scalar"):
        carried_loss.grad()(x, scratch)


def test_grad_submit_returns_a_future(gpu_backend):
    """The deferred spelling: one adjoint launch, gradients settle with the future."""
    @kernel
    def squared_error(x, target):
        total = 0.0
        for i in range(x.shape[0]):
            total += (x[i] - target[i]) ** 2
        return total

    x = _tracked()
    target = gnp.zeros(8, np.float32)
    future = squared_error.grad()[gpu_backend].submit(x, target)
    future.result()
    # The gradient buffers travel as the launch arguments; `.grad` holds them, and the
    # future resolving is what makes them safe to read.
    np.testing.assert_allclose(_host(x.grad), 2.0 * _host(x), rtol=1e-5)


def test_value_and_grad_refuses_submit():
    """`value_and_grad` answers the loss VALUE, which a deferred launch cannot carry."""
    @kernel
    def squared_error(x, target):
        total = 0.0
        for i in range(x.shape[0]):
            total += (x[i] - target[i]) ** 2
        return total

    x = _tracked()
    target = gnp.zeros(8, np.float32)
    with pytest.raises(SyntaxError, match="deferred"):
        squared_error.value_and_grad().submit(x, target)


def test_vjp_refuses_submit():
    """vjp's forward is synchronous and its pullback is a closure: no deferred spelling."""
    @kernel
    def twice(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0
        return out

    x = _tracked()
    out = gnp.zeros(8, np.float32)
    with pytest.raises(SyntaxError, match="deferred"):
        twice.vjp().submit(x, out)


def test_tape_still_records_around_deferred_calls(gpu_backend):
    """The tape records plain launches; an adjoint launch inside is never recorded."""
    @kernel
    def twice(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0
        return out

    @kernel
    def dot_loss(out, loss):
        for i in range(out.shape[0]):
            loss[0] += out[i] * out[i]

    @kernel
    def sse(x, loss):
        for i in range(x.shape[0]):
            loss[0] += x[i] * x[i]

    x = _tracked()
    out = gnp.zeros(8, np.float32)
    loss = gnp.zeros(1, np.float32)
    with Tape() as tape:
        twice[gpu_backend](x, out)
        dot_loss[gpu_backend](out, loss)
        x.zero_grad()
        sse.grad(loss="loss")(x, loss)          # an adjoint launch inside the tape
    assert len(tape.launches) == 2
    np.testing.assert_allclose(_host(x.grad), 2.0 * _host(x), rtol=1e-5)
