# Every named AD refusal, checked at build time with its position and its message.
#
# The plan's rule: anything the device cannot differentiate is refused at translation
# time with a `SyntaxError` carrying the line and the replacement spelling -- never a
# silent zero gradient. Each test below is one entry of that list.
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu.numpy as gnp
from pygpgpu import atomic_add, kernel


@pytest.fixture
def tracked():
    x = gnp.ones(8, np.float32)
    x.requires_grad = True
    return x


def _weights(n:int=8)->gnp.ndarray:
    return gnp.ones(n, np.float32)


def test_scatter_index_refused_naming_the_milestone(tracked):
    """A gradient whose index is not the work item's own is a scatter: refused in v1."""
    idx = gnp.zeros(8, np.int32)
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, idx, loss):
        for i in range(x.shape[0]):
            atomic_add(loss, 0, x[idx[i]] * 2.0)

    with pytest.raises(SyntaxError, match="atomic-scatter milestone"):
        body.grad(loss="loss")(tracked, idx, loss)


def test_wrt_empty_tuple_refused(tracked):
    """`wrt=()` is a derivative with respect to nothing: a mistake, not an empty answer."""
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, loss):
        for i in range(x.shape[0]):
            atomic_add(loss, 0, x[i] * x[i])

    with pytest.raises(SyntaxError, match="with respect to nothing"):
        body.grad(loss="loss", wrt=())(tracked, loss)


def test_positional_configuring_call_refused(tracked):
    """`k.grad(x, out)` tries to configure positionally: refused, naming the two calls."""
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, loss):
        for i in range(x.shape[0]):
            atomic_add(loss, 0, x[i] * x[i])

    with pytest.raises(SyntaxError, match="configuring call takes keywords only"):
        body.grad(tracked, loss)


def test_empty_wrt_inference_refused(tracked):
    """No flag and no name: the inference answers nothing and the call is refused."""
    plain = gnp.ones(8, np.float32)
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, loss):
        for i in range(x.shape[0]):
            atomic_add(loss, 0, x[i] * x[i])

    with pytest.raises(SyntaxError, match="requires_grad"):
        body.grad()(plain, loss)


def test_body_returning_nothing_refused(tracked):
    """No return and no `loss=`: both spellings are named in the refusal."""
    out = gnp.zeros(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0

    with pytest.raises(SyntaxError, match='loss="name"'):
        body.grad()(tracked, out)


def test_per_work_item_return_refused(tracked):
    """A return the launch does not share cannot be the loss."""
    out = gnp.zeros(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * x[i]
        return out[i]

    with pytest.raises(SyntaxError, match='loss="name"'):
        body.grad()(tracked, out)


def test_in_place_write_into_tracked_refused(tracked):
    """The adjoint needs the pre-state a store overwrites: refused by name."""
    out = gnp.zeros(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            x[i] = x[i] * 2.0
            out[i] = x[i]
        return out

    # The adjoint kernel is built when the pullback runs, so the refusal fires there.
    _result, pullback = body.vjp()(tracked, out)
    with pytest.raises(SyntaxError, match="requires_grad"):
        pullback(_weights())


def test_non_loss_atomic_refused(tracked):
    """The only supported atomic is `atomic_add` into the `loss=` parameter."""
    out = gnp.zeros(8, np.int32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            atomic_add(out, i, 1)
        return out

    _result, pullback = body.vjp()(tracked, out)
    with pytest.raises(SyntaxError, match="only supported atomic"):
        pullback(_weights())


def test_integer_wrt_refused_in_reverse(tracked):
    """A gradient through an integer array is not representable."""
    ints = gnp.ones(8, np.int32)
    ints.requires_grad = True
    out = gnp.zeros(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 1.0
        return out

    _result, pullback = body.vjp()(ints, out)
    with pytest.raises(SyntaxError, match="not representable"):
        pullback(_weights())


def test_complex_wrt_refused_by_name(tracked):
    """Complex AD needs the Wirtinger convention settled: refused, pointing at the split."""
    complex_x = gnp.ones(4, np.complex64)
    complex_x.requires_grad = True
    out = gnp.zeros(4, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i].real
        return out

    _result, pullback = body.vjp()(complex_x, out)
    with pytest.raises(SyntaxError, match="Wirtinger"):
        pullback(_weights(4))


def test_device_helper_without_adjoint_refused(tracked):
    """A helper the walk cannot differentiate is a dead end until one is registered."""
    out = gnp.zeros(8, np.float32)

    def my_scale(v):
        return v * 2.0

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = my_scale(x[i])
        return out

    _result, pullback = body.vjp()(tracked, out)
    with pytest.raises(SyntaxError, match="grad_for"):
        pullback(_weights())


def test_host_region_read_of_tracked_array_refused(tracked):
    """A host statement computing from a tracked array has no adjoint in v1."""
    out = gnp.zeros(8, np.float32)
    scale = gnp.ones(8, np.float32)
    scale.requires_grad = True

    @kernel
    def body(x, scale, out):
        shifted = scale * 2.0  # noqa: F841  (the host read the refusal is about)
        for i in range(out.shape[0]):
            out[i] = x[i] * 1.0
        return out

    _result, pullback = body.vjp()(tracked, scale, out)
    with pytest.raises(SyntaxError, match="host has no AD"):
        pullback(_weights())

def test_dispatch_loop_else_refused_in_reverse(tracked):
    """The dispatch loop's own `else` block never passes through the walk (the walk
    starts at the dispatch body), so `build` refuses it before walking."""
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, loss):
        for i in range(x.shape[0]):
            loss[0] += x[i]
        else:
            loss[0] += 0.0

    with pytest.raises(SyntaxError, match="`else` block"):
        body.grad(loss="loss")(tracked, loss)


def test_nested_while_else_refused_in_reverse(tracked):
    """A nested loop's `else` block is refused by the walk itself, at the loop."""
    loss = gnp.zeros(1, np.float32)

    @kernel
    def body(x, loss):
        for i in range(x.shape[0]):
            v = x[i]
            while v > 1.0:
                v = v * 0.5
            else:
                loss[0] += v

    with pytest.raises(SyntaxError, match="`else` block"):
        body.grad(loss="loss")(tracked, loss)


def test_loop_else_refused_in_forward_mode(tracked):
    """The forward transform refuses the same shape: the else block gains no tangent."""
    out = gnp.zeros(8, np.float32)

    @kernel
    def body(x, out):
        for i in range(out.shape[0]):
            out[i] = x[i] * 2.0
        else:
            out[0] = 0.0

    form = body.jvp(tangent=gnp.ones(8, np.float32), out="out")["python"]
    with pytest.raises(SyntaxError, match="`else` block"):
        form(tracked, out)
