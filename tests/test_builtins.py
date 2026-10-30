"""The device builtins are plain top-level functions with a written-out signature each.

There is no `gpu` namespace: every builtin is exported from `pygpgpu` directly, so an
editor completes `pygpgpu.` and a type checker sees the real parameters. Each one is
*dual-sided* -- the device backends recognize it by the object the callee resolves to and
emit the matching call, never reading the body, while the python backend runs that body.
These tests hold the two sides together: a name the translator recognizes must be exported
with a real signature, and calling it on the host must do what the device call means.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import pygpgpu
from pygpgpu import (
    atomic_add,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_sub,
    atomic_xchg,
    barrier,
    clamp,
    dot,
    get_global_id,
    kernel,
    local,
    work_size,
)
from pygpgpu.kernel import builtins as namespace
from pygpgpu.kernel import device_builtins as builtin_tables
from pygpgpu.kernel.builtins import dot as imported_dot


def _dispatched_names() -> list[str]:
    """Every name the translator dispatches on."""
    return sorted(set(builtin_tables.INDEX_BUILTINS)
                  | set(builtin_tables.ATOMIC_BUILTINS)
                  | set(builtin_tables.GENERIC_BUILTINS)
                  | {"local", "work_size"})


# ---------------------------------------------------------------------------
# Shape of the API
# ---------------------------------------------------------------------------


def test_there_is_no_gpu_namespace_any_more():
    """The builtins are individual names, not attributes of a namespace object."""
    assert not hasattr(pygpgpu, "gpu")
    assert "gpu" not in pygpgpu.__all__


def test_every_dispatched_name_is_exported_at_the_top_level():
    missing: list[str] = [name for name in _dispatched_names()
                          if not hasattr(pygpgpu, name)]
    assert not missing, f"translator recognizes these but pygpgpu does not export them: {missing}"


def test_every_dispatched_name_is_in_the_public_list():
    exported = set(pygpgpu.__all__)
    missing: list[str] = [name for name in _dispatched_names() if name not in exported]
    assert not missing, f"names missing from pygpgpu.__all__: {missing}"


def test_the_module_and_the_package_expose_the_same_objects():
    for name in namespace.__all__:
        assert getattr(pygpgpu, name) is getattr(namespace, name), name


# These two take no fixed argument list by design: barrier() has nothing to wait on, and
# work_size() accepts any number of arrays to declare the domain.
_VARIADIC_BY_DESIGN: set[str] = {"barrier", "work_size"}


@pytest.mark.parametrize("name", _dispatched_names())
def test_each_builtin_has_a_real_signature(name: str):
    """A stub would answer with ``(*args, **kwargs)``; the whole point is not to."""
    function = getattr(pygpgpu, name)
    assert callable(function)
    signature = inspect.signature(function)
    if name in _VARIADIC_BY_DESIGN:
        return
    for parameter in signature.parameters.values():
        assert parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD), (
            f"{name} still takes *args/**kwargs, so an editor cannot show its arguments")




@pytest.mark.parametrize("name", ["get_global_id", "atomic_cas", "normalize", "work_size"])
def test_parameter_names_are_meaningful(name: str):
    signature = inspect.signature(getattr(pygpgpu, name))
    assert "args" not in signature.parameters
    assert "kwargs" not in signature.parameters


# ---------------------------------------------------------------------------
# The host side actually implements the operation
# ---------------------------------------------------------------------------


def test_atomics_work_on_a_host_array():
    out = np.zeros(3, dtype=np.int32)
    assert atomic_add(out, 0, 5) == 0
    assert out[0] == 5
    assert atomic_sub(out, 0, 2) == 5
    assert out[0] == 3
    assert atomic_min(out, 0, 7) == 3
    assert out[0] == 3
    assert atomic_max(out, 0, 7) == 3
    assert out[0] == 7
    assert atomic_xchg(out, 0, 1) == 7
    assert out[0] == 1


def test_cas_reports_and_controls_the_swap():
    out = np.array([0], dtype=np.int32)
    assert atomic_cas(out, 0, 0, 9) == 0
    assert out[0] == 9
    assert atomic_cas(out, 0, 0, 4) == 9  # compare failed, value unchanged
    assert out[0] == 9


def test_a_flat_index_addresses_a_multi_dimensional_buffer():
    """The device call indexes a raw pointer, so the rank must not matter here either."""
    out = np.zeros((2, 3), dtype=np.int32)
    atomic_add(out, 5, 1)
    assert out[1, 2] == 1


def test_work_item_builtins_explain_themselves_outside_a_kernel():
    with pytest.raises(SyntaxError) as caught:
        get_global_id(0)
    assert "inside a @kernel" in str(caught.value)


def test_work_size_and_barrier_are_no_ops_on_the_host():
    assert work_size(np.zeros(4, dtype=np.float32)) is None
    assert barrier() is None


def test_local_is_refused_on_the_host():
    with pytest.raises(SyntaxError) as caught:
        local(np.float32, 16)
    assert "local" in str(caught.value)


# ---------------------------------------------------------------------------
# A builtin is recognized by the object it resolves to, not by its name
# ---------------------------------------------------------------------------


def user_dot(a, b):
    """Shares a builtin's name on purpose: doubled instead of dotted, so a mix-up shows."""
    return a * 2.0 + b


def user_get_global_id(dim):
    """Also shares a builtin's name; must still be usable as an ordinary helper."""
    return dim + 100


@kernel
def calls_user_dot(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = user_dot(x[i], x[i])


@kernel
def calls_user_named_like_builtin(out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = user_get_global_id(i)


@kernel
def calls_imported_dot(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = imported_dot(x[i], x[i])


@kernel
def calls_namespace_dot(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = dot(x[i], x[i])


def test_a_helper_sharing_a_builtin_name_stays_a_helper(runner):
    x = np.array([3.0, 4.0], dtype=np.float32)
    out = np.zeros(2, dtype=np.float32)
    runner(calls_user_dot, x, out)
    np.testing.assert_allclose(out, x * 2.0 + x)


def test_a_helper_sharing_a_work_item_builtin_name_stays_a_helper(runner):
    out = np.zeros(2, dtype=np.float32)
    runner(calls_user_named_like_builtin, out)
    np.testing.assert_allclose(out, np.array([100.0, 101.0], dtype=np.float32))


def test_a_builtin_imported_by_name_still_translates_as_the_builtin(runner):
    """The same function object, reached by `from ... import dot`, is the builtin."""
    x = np.array([3.0, 4.0], dtype=np.float32)
    by_import = np.zeros(2, dtype=np.float32)
    by_namespace = np.zeros(2, dtype=np.float32)
    runner(calls_imported_dot, x, by_import)
    runner(calls_namespace_dot, x, by_namespace)
    np.testing.assert_allclose(by_import, by_namespace)
    np.testing.assert_allclose(by_import, x * x)


def test_the_lookup_is_by_object_identity():
    from pygpgpu.kernel.builtin_registry import builtin_name_of

    assert builtin_name_of(dot) == "dot"
    assert builtin_name_of(imported_dot) == "dot"
    assert builtin_name_of(user_dot) is None
    assert builtin_name_of(lambda a, b: a) is None


def test_an_attribute_that_is_not_a_builtin_falls_through(gpu_backend, runner):
    """Resolving to the package is not enough: the attribute must be a builtin object."""

    @kernel
    def calls_missing(out: np.ndarray):
        for i in range(out.shape[0]):
            out[i] = pygpgpu.not_a_builtin(i)

    with pytest.raises(SyntaxError):
        runner(calls_missing, np.zeros(1, dtype=np.float32))


# ---------------------------------------------------------------------------
# Both sides agree through a real launch
# ---------------------------------------------------------------------------


@kernel
def clamp_through_gpu(x: np.ndarray, out: np.ndarray):
    for i in range(out.shape[0]):
        out[i] = clamp(x[i], 0.0, 1.0)


@kernel
def atomic_through_gpu(x: np.ndarray, out: np.ndarray):
    for i in range(x.shape[0]):
        atomic_add(out, x[i], 1)


def test_a_vector_builtin_resolves_on_every_backend(runner):
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    runner(clamp_through_gpu, x, out)
    np.testing.assert_allclose(out, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_an_atomic_builtin_resolves_on_every_backend(runner):
    x = np.array([0, 1, 1, 2], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)
    runner(atomic_through_gpu, x, out)
    np.testing.assert_array_equal(out, np.array([1, 2, 1], dtype=np.int32))
