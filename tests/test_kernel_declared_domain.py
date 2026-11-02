"""`@kernel(domain=...)`: name the array whose shape declares the iteration domain.

A gid-style body has no magic loop for the launch to read a domain from, so it must say where
the domain comes from. `domain=` is part of the **signature**, where that fact belongs -- it
emits no device code, because it only decides the launch geometry. This module pins the
mechanism (1-D and 2-D domains, sizes that differ per call, a rank wider than a grid) and the
three mistakes the string-valued argument invites: a name that is not a parameter, a parameter
that is not an array, and a body that already declares a domain with a magic loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from pygpgpu import kernel, get_global_id


@kernel(domain="x")
def flat_copy(x: np.ndarray, out: np.ndarray):
    """The usual 1-D reduction shape: one work item per element."""
    gid = get_global_id(0)
    out[gid] = x[gid]


@kernel(domain="x")
def grid_copy(x: np.ndarray, out: np.ndarray):
    """A two-axis domain: both ids are real, which is what a 2-D grid gives."""
    i = get_global_id(0)
    j = get_global_id(1)
    out[i, j] = x[i, j]


def test_the_decorator_declares_the_domain(gpu_backend):
    x = np.arange(16, dtype=np.float32)
    out = np.zeros(16, dtype=np.float32)
    flat_copy[gpu_backend](x, out)
    np.testing.assert_array_equal(out, x)


def test_a_two_axis_domain_maps_both_ids(gpu_backend):
    shape = (4, 5)
    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    out = np.zeros(shape, dtype=np.float32)
    grid_copy[gpu_backend](x, out)
    np.testing.assert_array_equal(out, x)


def test_the_domain_is_read_per_call_not_frozen(gpu_backend):
    """The declaration is a *rule*, not a size: each call derives from its own argument."""
    for size in (8, 100, 17, 8):
        x = np.ones(size, dtype=np.float32)
        out = np.zeros(size, dtype=np.float32)
        flat_copy[gpu_backend](x, out)
        np.testing.assert_array_equal(out, x)


def test_a_domain_that_names_no_parameter_is_refused(gpu_backend):
    """A string argument does not follow a rename, so the mistake has to be reported."""
    @kernel(domain="nope")
    def wrong_name(x: np.ndarray, out: np.ndarray):
        out[get_global_id(0)] = x[0]

    with pytest.raises(SyntaxError) as caught:
        wrong_name[gpu_backend](np.zeros(4, dtype=np.float32),
                                np.zeros(4, dtype=np.float32))
    message: str = str(caught.value)
    assert "names no parameter" in message, message
    # The message lists what does exist, which is what a renamed parameter leaves behind.
    assert "x, out" in message, message


def test_a_domain_over_a_scalar_is_refused(gpu_backend):
    """Only an array carries the shape dimensions a launch is derived from."""
    @kernel(domain="n")
    def scalar_domain(n, out: np.ndarray):
        out[0] = n

    with pytest.raises(SyntaxError) as caught:
        scalar_domain[gpu_backend](1.0, np.zeros(4, dtype=np.float32))
    message: str = str(caught.value)
    assert "requires an array parameter" in message, message


def test_a_body_with_a_magic_loop_may_not_also_declare_a_domain(gpu_backend):
    """One declaration per kernel: the loop already fixes the domain."""
    @kernel(domain="x")
    def loop_and_domain(x: np.ndarray, out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = x[i]

    with pytest.raises(SyntaxError) as caught:
        loop_and_domain[gpu_backend](np.zeros(4, dtype=np.float32),
                                     np.zeros(4, dtype=np.float32))
    assert "magic loop" in str(caught.value), str(caught.value)


@kernel(domain="x")
def flat_position_over_five_axes(x: np.ndarray, out: np.ndarray):
    """A rank-5 domain folds to `prod(shape)` work items, so gid 0 is the flat position."""
    gid = get_global_id(0)
    out[gid, 0, 0, 0, 0] = x[gid, 0, 0, 0, 0]


def test_a_domain_of_more_than_three_axes_launches_flat(gpu_backend):
    """More axes than a grid has are folded into one flat axis, as the original code did.

    The launch therefore covers `prod(shape)` work items and `get_global_id(0)` is the flat
    position, so the ids the body does not iterate read 0. A body that wants the original
    axes spelled out iterates with `np.ndindex` instead -- see the traversal-rank tests.
    """
    shape = (2, 3, 2, 2, 3)
    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    out = np.zeros(shape, dtype=np.float32)
    flat_position_over_five_axes[gpu_backend](x, out)
    # The body writes `out[gid, 0, 0, 0, 0]`, and gid is the flat position over the whole
    # domain, so exactly the leading column is covered, in order.
    np.testing.assert_array_equal(out[:, 0, 0, 0, 0], x[:, 0, 0, 0, 0])


def _shared_body(x: np.ndarray, out: np.ndarray):
    gid = get_global_id(0)
    out[gid] = x[gid]


# One body, two decorators, two different domains. Both are legal and independent: the domain
# decides which array supplies the geometry, and geometry is a per-call host-side fact, so the
# pair shares a generated source (and a C name) without either launch borrowing the other's.
SIZED_BY_OUTPUT = kernel(domain="out")(_shared_body)
SIZED_BY_INPUT = kernel(domain="x")(_shared_body)


def test_two_domains_over_one_body_stay_independent(gpu_backend):
    """A call on either decorator launches from its own array, at whatever size that call has.

    The reusable fact is the compiled source, not the geometry: each `PyKernel` keeps its own
    `_cache`, and the size is read fresh from the argument on every call.
    """
    assert SIZED_BY_OUTPUT._source_sig == SIZED_BY_INPUT._source_sig
    assert SIZED_BY_OUTPUT._cache is not SIZED_BY_INPUT._cache

    for size in (8, 100, 17, 8):
        for decorated in (SIZED_BY_OUTPUT, SIZED_BY_INPUT):
            x = np.ones(size, dtype=np.float32)
            out = np.zeros(size, dtype=np.float32)
            decorated[gpu_backend](x, out)
            np.testing.assert_array_equal(out, x)
