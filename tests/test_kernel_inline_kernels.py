"""`with inline, kernel(...):` -- a `@kernel` body written where it is launched.

The body is the kernel, spelled exactly as it would be inside a `@kernel` function: the
same magic loop, the same three host regions, the same copy policy, the same generated
file and cache, the same backends. What differs is only where it lives:

* the `kernel(...)` call carries **options, not data** -- `default_device=`, `int=`, `float=`,
  and the work-size pair `blocks_per_grid=` / `threads_per_block=`;
* the data reaches the device as it does for a `@kernel` body: a name the block reads is a
  parameter (in a decorated kernel a parameter or a closure cell, here the enclosing
  function's local), and a module global stays a module global.

The domain is the **magic loop's**: `for i in range(<array>.shape[0])`. A gid-style body is a
`@kernel` spelling (its domain has to be named away from the body, which a block written at the
launch site never needs), so it is refused here by name.

The caller's own frame never executes a line of the body: entering the `with` head raises
an internal signal that `inline.__exit__` catches, and the catch launches the block.

The refusals live in `test_kernel_inline_kernels_refusals.py`, and error positions plus
block placement in `test_kernel_inline_kernels_placement.py`.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

import pygpgpu
from pygpgpu import get_global_id, inline, kernel
from pygpgpu.kernel.inline_kernels.launch import _SITES


def _close(actual:object, expected:object)->None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


_GLOBAL_FACTOR:float = 3.0
_MODULE_COUNTER:int = 0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_inline_is_exported_from_both_namespaces()->None:
    assert pygpgpu.inline is inline
    assert "inline" in pygpgpu.__all__
    package:object = sys.modules["pygpgpu.kernel"]
    assert package.inline is inline                     # type: ignore[attr-defined]
    assert "inline" in package.__all__                  # type: ignore[attr-defined]


def test_kernel_still_returns_a_decorator_factory()->None:
    """`kernel(options)` serves both roles: decorator factory and block head."""
    spec = kernel(default_device="python")
    assert callable(spec)

    @spec
    def add(a, out):
        for i in range(out.shape[0]):
            out[i] = a[i] + 1.0

    data:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    add(data, out)

    _close(out, data + 1.0)


# ---------------------------------------------------------------------------
# The launch itself
# ---------------------------------------------------------------------------


def test_an_element_wise_block_writes_the_captured_array(backend)->None:
    a:np.ndarray = np.arange(8, dtype=np.float32)
    b:np.ndarray = np.arange(8, dtype=np.float32) * 2.0
    out:np.ndarray = np.zeros(8, dtype=np.float32)
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * 2.0 + b[i]
    _close(out, a * 2.0 + b)


def test_the_caller_never_executes_the_block(backend)->None:
    """The body is device code: the caller's own frame never runs a line of it."""
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = 1.0
    # If the body had run in this frame, the loop name would be a local of it now.
    assert "i" not in locals()
    _close(out, np.ones(4))


def test_an_uncaptured_module_global_is_read(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * _GLOBAL_FACTOR
    _close(out, a * _GLOBAL_FACTOR)


def test_a_local_shadowing_a_module_global_wins(backend)->None:
    """The block sees the enclosing local, exactly as the same body in a `def` would."""
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    _GLOBAL_FACTOR = 10.0                        # noqa: F841 - shadows the module global
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * _GLOBAL_FACTOR
    _close(out, a * 10.0)


def test_the_capture_order_follows_the_source(backend)->None:
    """Two captures keep the order the body reads them in, so the launch is followable."""
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    first:np.ndarray = np.arange(4, dtype=np.float32)
    second:np.ndarray = np.arange(4, dtype=np.float32) * 3.0
    seen:list = []
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = first[i] + second[i]
    seen.append(list(_SITES.values())[-1].names)
    # `out` is read first (`out.shape`), then `first`, then `second` -- body order.
    assert seen[0] == ["out", "first", "second"], seen[0]
    _close(out, first + second)


def test_a_captured_scalar_travels_as_a_parameter(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    factor:float = 2.5
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i] * factor
    _close(out, a * 2.5)


def test_a_captured_scalar_bounds_a_serial_loop(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    count:int = 3
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            total:float = 0.0
            for j in range(count):
                total = total + a[i]
            out[i] = total
    _close(out, a * 3.0)


def test_a_two_pass_block_writes_both_phases(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    mid:np.ndarray = np.zeros(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device=backend):
        for i in range(mid.shape[0]):
            mid[i] = a[i] + 1.0
        for i in range(out.shape[0]):
            out[i] = mid[i] * 2.0
    _close(mid, a + 1.0)
    _close(out, (a + 1.0) * 2.0)


def test_the_host_regions_of_a_block(backend)->None:
    """Above the loop runs once on the host; the tail runs after the launch."""
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    flag:np.ndarray = np.zeros(1, dtype=np.float32)
    with inline, kernel(default_device=backend):
        width:int = a.shape[0]
        for i in range(out.shape[0]):
            out[i] = a[i] + width
        flag[0] = float(out[0])
    _close(out, a + 4.0)
    assert flag[0] == pytest.approx(float(a[0]) + 4.0)


def test_a_global_written_in_a_block_reaches_the_module(backend)->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device=backend):
        global _MODULE_COUNTER
        for i in range(out.shape[0]):
            out[i] = float(i)
        _MODULE_COUNTER = 7
    assert _MODULE_COUNTER == 7


def test_a_block_calls_a_device_helper(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = _twice(a[i])
    _close(out, a * 2.0)


def test_a_decorated_kernel_is_not_a_device_helper(gpu_backend)->None:
    """A decorated kernel called from a block is a launch, exactly as from a `@kernel` body.

    The interpreter backend runs the body natively, so there a nested launch is what python
    means; the device backends translate it, where a `@kernel` function is not a helper.
    """
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="is a @kernel function"):
        with inline, kernel(default_device=gpu_backend):
            for i in range(out.shape[0]):
                out[i] = _decorated_twice(a[i])


@kernel
def _decorated_twice(value:float)->float:
    return value + value


def _twice(value:float)->float:
    return value + value


def test_a_return_in_a_block_answers_through_the_as_target(backend)->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    a:np.ndarray = np.arange(4, dtype=np.float32)
    with inline as result, kernel(default_device=backend):
        for i in range(out.shape[0]):
            out[i] = a[i]
        return 42
    _close(out, a)
    assert result.value == 42


# ---------------------------------------------------------------------------
# Launching a child kernel from inside a block
# ---------------------------------------------------------------------------


@kernel
def _double_child(a:np.ndarray, out:np.ndarray)->None:
    for i in range(out.shape[0]):
        out[i] = a[i] * 2.0


def _require_cuda()->None:
    """Skip unless this machine has CUDA: a device-side launch needs dynamic parallelism."""
    if "cuda:0" not in pygpgpu.list_devices():
        pytest.skip("a device-side launch needs CUDA's dynamic parallelism")


def test_a_block_launches_a_child_kernel()->None:
    """The device-side launch a block can perform: CUDA's dynamic parallelism.

    A body that launches a child delegates the work division to it, so it needs no magic
    loop of its own -- the launch IS its work size. This is the same `child[(n,)](...)`
    spelling a `@kernel` body uses, which is what makes the block and the decorator one
    execution model rather than two.

    CUDA-specific: a device-side launch needs dynamic parallelism, which is CUDA's alone
    (OpenCL's kernel language has no nested-launch statement), so this follows
    `test_kernel_nested_launch.py` in naming the backend instead of using `gpu_backend`.
    """
    _require_cuda()
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with inline, kernel(default_device="cuda"):
        _double_child[(out.shape[0],)](a, out)
    _close(out, a * 2.0)


def test_a_child_kernel_launched_from_a_block_gets_its_extent()->None:
    """A magic-loop child derives extent parameters; the launch has to forward them.

    This is the case that used to be a C compile error ("too few arguments in function
    call"): the child's signature ends with `int _out_shape0`, and a device-side launch that
    omitted it cannot compile. It is pinned here because a block is a natural place to write
    exactly this shape.
    """
    _require_cuda()
    a:np.ndarray = np.arange(6, dtype=np.float32)
    out:np.ndarray = np.zeros(6, dtype=np.float32)
    with inline, kernel(default_device="cuda"):
        _double_child[(out.shape[0],)](a, out)
    _close(out, a * 2.0)


def test_a_backend_without_device_side_launch_says_so()->None:
    """OpenCL has no nested-launch statement, so the block reports it by backend name."""
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="dynamic parallelism"):
        with inline, kernel(default_device="opencl"):
            _double_child[(out.shape[0],)](a, out)


# ---------------------------------------------------------------------------
# Work size
# ---------------------------------------------------------------------------


def test_threads_per_block_sets_the_group_width(backend)->None:
    a:np.ndarray = np.arange(8, dtype=np.float32)
    out:np.ndarray = np.zeros(8, dtype=np.float32)
    with inline, kernel(default_device=backend, threads_per_block=4):
        for i in range(out.shape[0]):
            out[i] = a[i] + 1.0
    _close(out, a + 1.0)


def test_blocks_and_threads_together_set_the_launch(backend)->None:
    a:np.ndarray = np.arange(8, dtype=np.float32)
    out:np.ndarray = np.zeros(8, dtype=np.float32)
    with inline, kernel(default_device=backend, blocks_per_grid=2, threads_per_block=4):
        for i in range(out.shape[0]):
            out[i] = a[i] + 1.0
    _close(out, a + 1.0)


def test_the_magic_loop_is_the_only_way_a_block_states_its_domain(gpu_backend)->None:
    """A gid-style body is a `@kernel` spelling: a block must carry its own magic loop."""
    out:np.ndarray = np.zeros(6, dtype=np.float32)
    with pytest.raises(SyntaxError, match="gid-style"):
        with inline, kernel(default_device=gpu_backend):
            lane:int = get_global_id(0)
            if lane < out.shape[0]:
                out[lane] = float(lane) + 1.0


def test_a_block_with_no_loop_and_no_device_call_is_refused()->None:
    """Without a magic loop there is no work size, so the body has to say one."""
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="magic loop"):
        with inline, kernel(default_device="python"):
            out[0] = 1.0


def test_a_domain_option_on_a_block_is_refused()->None:
    """`domain=` is the decorator's spelling; the block's own loop states the domain."""
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    with pytest.raises(SyntaxError, match="domain"):
        with inline, kernel(default_device="python", domain="out"):
            for i in range(out.shape[0]):
                out[i] = 1.0


# ---------------------------------------------------------------------------
# One kernel per site
# ---------------------------------------------------------------------------


def test_a_site_executed_twice_builds_one_kernel(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    before:int = len(_SITES)
    for _ in range(2):
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = out[i] + a[i] + 1.0
    assert len(_SITES) == before + 1
    _close(out, a * 2.0 + 2.0)


def test_two_sites_in_one_function_are_both_launched(backend)->None:
    a:np.ndarray = np.arange(4, dtype=np.float32)
    first:np.ndarray = np.zeros(4, dtype=np.float32)
    second:np.ndarray = np.zeros(4, dtype=np.float32)
    before:int = len(_SITES)
    with inline, kernel(default_device=backend):
        for i in range(first.shape[0]):
            first[i] = a[i] + 1.0
    with inline, kernel(default_device=backend):
        for i in range(second.shape[0]):
            second[i] = a[i] + 2.0
    assert len(_SITES) == before + 2
    _close(first, a + 1.0)
    _close(second, a + 2.0)


def test_two_executions_of_one_site_share_the_kernel_object(backend)->None:
    out:np.ndarray = np.zeros(4, dtype=np.float32)
    seen:list = []
    for _ in range(2):
        with inline, kernel(default_device=backend):
            for i in range(out.shape[0]):
                out[i] = 1.0
        seen.append(list(_SITES.values())[-1].kernel)
    assert seen[0] is seen[1]
