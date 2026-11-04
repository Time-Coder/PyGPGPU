"""The gnp transfer API: device residency, async copies, and cross-device staging.

These tests pin the behaviour ``pygpgpu.numpy`` exists for, and the rules the engine
follows when a copy cannot simply be skipped:

* a kernel call does not bring the data home; the next host read does;
* a second launch on the same device moves nothing;
* a launch on another device (or another backend) stages through the host, and the
  result there is the newest one, not whichever buffer happened to be current;
* ``to_device``/``to_host`` are asynchronous, and the points that depend on their result
  wait for them;
* zero-copy is offered where the API guarantees it and refused, by name, where it does not.
"""
from __future__ import annotations

import numpy as np
import pytest

import pygpgpu
import pygpgpu.numpy as gnp
from pygpgpu import kernel


@kernel
def saxpy(x: np.ndarray, y: np.ndarray, out: np.ndarray, alpha: float) -> None:
    for i in range(out.shape[0]):
        out[i] = x[i] * alpha + y[i]


@kernel
def bump(x: np.ndarray) -> None:
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0


@kernel
def total_of(x: np.ndarray, out: np.ndarray) -> None:
    for i in range(x.shape[0]):
        out[0] += x[i]


def _bindings(array) -> dict:
    """The device copies an array currently holds, keyed by (backend, device)."""
    return {(key[0][0], key[0][1]): binding for key, binding in array._bindings.items()}


def _device_name(target_key) -> str:
    return target_key[1].name if hasattr(target_key[1], "name") else str(target_key[1])


# ---------- construction ----------

def test_constructors_return_owned_arrays():
    """Every constructor allocates, so a gnp array's family *is* its own memory."""
    for make in (lambda: gnp.zeros(4, np.float32), lambda: gnp.arange(4),
                 lambda: gnp.empty(4, np.float32), lambda: gnp.ones(4)):
        array = make()
        assert isinstance(array, gnp.ndarray)
        # `live` means the family's bytes alias the array itself, so no staging copy is
        # ever made for it -- the ownership fact that matters to the copy engine.
        family = array._state
        assert family.live, "a constructor's array must not need a staging copy"
        assert np.shares_memory(family.data, np.asarray(array))


def test_arithmetic_and_reductions_are_ordinary_numpy():
    """The whole point is that a gnp array *is* a numpy array on the host."""
    a = gnp.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(np.asarray(a + 1), np.arange(12, dtype=np.float32).reshape(3, 4) + 1)
    assert float(np.sum(a)) == 66.0
    assert float(a.mean()) == 5.5
    assert float(a[1, 2]) == 6.0
    assert a.tolist()[0] == [0.0, 1.0, 2.0, 3.0]
    assert (a @ a.T).shape == (3, 3)


def test_a_view_stays_in_the_family():
    a = gnp.arange(12, dtype=np.float32).reshape(3, 4)
    view = a[1]
    assert view.base is a
    # A reshape is a view of a view, so the chain ends at the same family root.
    assert view._root() is a._root()
    assert view._state is a._state
    assert a.T.base is a


# ---------- residency ----------

def test_a_launch_leaves_the_data_on_the_device(gpu_backend):
    """Nothing is copied home by a launch: the host bytes stay stale until they are read."""
    a = gnp.arange(64, dtype=np.float32)
    b = gnp.arange(64, dtype=np.float32)
    out = gnp.zeros(64, dtype=np.float32)

    saxpy[gpu_backend](a, b, out, 2.0)

    family = out._state
    assert family.current_epoch < family.epoch, (
        "a written array must be marked stale on the host after the launch")
    assert any(binding.epoch == family.epoch for binding in _bindings(out).values())


def test_a_second_launch_on_one_device_moves_nothing(gpu_backend):
    """The resident buffer is reused, so the second call has no upload at all."""
    a = gnp.arange(1024, dtype=np.float32)
    b = gnp.arange(1024, dtype=np.float32)
    out = gnp.zeros(1024, dtype=np.float32)

    saxpy[gpu_backend](a, b, out, 2.0)
    resident = _bindings(a)[(gpu_backend, pygpgpu.get_device(gpu_backend))]

    saxpy[gpu_backend](a, b, out, 2.0)

    same = _bindings(a)[(gpu_backend, pygpgpu.get_device(gpu_backend))]
    assert same.mem_obj is resident.mem_obj, "the buffer must be reused, not re-allocated"
    assert same.epoch == a._state.epoch


def test_the_host_read_pulls_the_device_result_home(gpu_backend):
    a = gnp.arange(64, dtype=np.float32)
    b = np.full(64, 1.0, dtype=np.float32)
    out = gnp.zeros(64, dtype=np.float32)

    saxpy[gpu_backend](a, b, out, 2.0)

    expected = np.arange(64, dtype=np.float32) * 2 + 1
    np.testing.assert_allclose(out.tolist(), expected.tolist())
    assert out._state.current_epoch == out._state.epoch


def test_in_place_kernel_keeps_the_update_on_the_device(gpu_backend):
    """A read/write argument is uploaded once and then stays resident."""
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)
    assert float(a[0]) == 1.0
    bump[gpu_backend](a)
    assert float(a[0]) == 2.0


def test_a_host_write_invalidates_every_device_copy(gpu_backend):
    """The epoch bump is what makes 'the caller changed it' safe on every device."""
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)
    assert float(a[0]) == 1.0

    a[0] = 100.0
    bump[gpu_backend](a)
    assert float(a[0]) == 101.0, "the device must have seen the host write"


def test_two_separate_views_of_one_buffer_are_kernel_arguments(gpu_backend):
    """`k(a[0], a[0], a[0])` names three *different* view objects over one buffer.

    The launch binds that buffer once (``seen_arrays``), so the binding is stored on
    whichever view bound it first. The write-back arrives through the third view, which
    holds no binding of its own -- looking only at its own ``_bindings`` found nothing
    and silently dropped the result, leaving the caller's array unchanged.
    """
    grid = gnp.zeros((4, 4), dtype=np.float32)
    grid[0] = 1.0

    row_in, row_b, row_out = grid[0], grid[0], grid[0]
    assert row_in is not row_out, "three subscripts are three separate view objects"

    saxpy[gpu_backend](row_in, row_b, row_out, 2.0)

    assert row_out.tolist() == [3.0, 3.0, 3.0, 3.0]
    assert grid[0].tolist() == [3.0, 3.0, 3.0, 3.0], "the parent must see it too"


def test_a_non_contiguous_view_is_a_kernel_argument(gpu_backend):
    """A strided view must reach the device as *its own* bytes, and come back home.

    The backend converts a plain ndarray to a contiguous copy before uploading, which for
    a gnp array would hand the launch a copy and orphan the array's device buffer -- the
    kernel's writes would then never reach the caller. A gnp argument has to keep its own
    object all the way through the launch.
    """
    base = gnp.arange(16, dtype=np.float32).reshape(4, 4)
    column = base[:, 0]
    assert not column.flags["C_CONTIGUOUS"], "this test needs a strided view"

    bump[gpu_backend](column)

    assert float(column[0]) == 1.0, "the view must see the device's write"
    assert float(base[0, 0]) == 1.0, "and the write must reach the view's parent"
    assert float(base[1, 0]) == 5.0, "untouched rows keep their values"


def test_a_contiguous_view_is_a_kernel_argument(gpu_backend):
    """A contiguous view is a window into the parent, so it needs no private copy."""
    base = gnp.arange(16, dtype=np.float32).reshape(4, 4)
    row = base[0]
    assert row.flags["C_CONTIGUOUS"]

    bump[gpu_backend](row)

    assert float(row[0]) == 1.0
    assert float(base[0, 0]) == 1.0


# ---------- explicit transfers ----------

def test_to_device_returns_a_transfer_that_can_be_awaited(gpu_backend):
    a = gnp.arange(64, dtype=np.float32)
    transfer = a.to_device(gpu_backend)
    assert isinstance(transfer, gnp.Transfer)
    transfer.wait()
    assert transfer.done
    assert all(binding.epoch == a._state.epoch for binding in a._bindings.values())


def test_a_current_array_needs_no_copy(gpu_backend):
    """Asking for a device it is already on answers a completed transfer."""
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)          # a is now resident
    second = a.to_device(gpu_backend)
    assert second.done, "an array already current on the device must not copy again"
    assert len(_bindings(a)) == 1


def test_to_host_is_asynchronous_but_the_next_read_waits(gpu_backend):
    """An un-awaited ``to_host`` is still correct, because a read waits for it."""
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)          # device holds a + 1, host bytes are stale

    transfer = a.to_host()
    assert isinstance(transfer, gnp.Transfer)
    # Not necessarily done yet -- but this read must not see the stale bytes.
    assert float(a[0]) == 1.0
    transfer.wait()


def test_wait_through_a_future_gives_the_same_guarantee(gpu_backend):
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)
    future = a.to_host().future()
    future.result()
    assert float(a[0]) == 1.0


# ---------- cross-device ----------

def _second_backend(backend: str):
    """A backend other than this one that has a device, as a ``backend:0`` selector."""
    for name in ("cuda", "hip", "opencl"):
        if name != backend and pygpgpu.list_devices().count(f"{name}:0"):
            return f"{name}:0"
    return None


def test_another_backend_stages_through_the_host(gpu_backend):
    """A cross-backend transfer copies host -> host -> other device, and stays correct."""
    other = _second_backend(gpu_backend)
    if other is None:
        pytest.skip("needs a second backend")

    a = gnp.arange(64, dtype=np.float32)
    b = gnp.full(64, 1.0, dtype=np.float32)
    out = gnp.zeros(64, dtype=np.float32)

    saxpy[gpu_backend](a, b, out, 2.0)          # write it on the first backend
    saxpy[other](a, b, out, 3.0)                # read the newest content there

    expected = np.arange(64, dtype=np.float32) * 3 + 1
    np.testing.assert_allclose(out.tolist(), expected.tolist())
    assert out._state.epoch == out._state.current_epoch


def test_the_newest_device_wins_after_two_writes(gpu_backend):
    """A write on the second backend must be the one a host read pulls home."""
    other = _second_backend(gpu_backend)
    if other is None:
        pytest.skip("needs a second backend")

    a = gnp.arange(64, dtype=np.float32)
    b = np.full(64, 1.0, dtype=np.float32)
    out = gnp.zeros(64, dtype=np.float32)

    saxpy[gpu_backend](a, b, out, 2.0)
    saxpy[other](a, b, out, 5.0)

    family = out._state
    newest = family.newest_binding()
    assert newest is not None
    assert newest[1].epoch == family.epoch
    np.testing.assert_allclose(
        out.tolist(), (np.arange(64, dtype=np.float32) * 5 + 1).tolist())


# ---------- zero-copy ----------

@pytest.mark.parametrize("backend", ["opencl"])
def test_opencl_refuses_zero_copy_by_name(backend):
    a = gnp.arange(64, dtype=np.float32)
    supported, reason = gnp.zero_copy_support(f"{backend}:0")
    assert not supported
    assert "CL_MEM_USE_HOST_PTR" in reason, reason
    with pytest.raises(NotImplementedError, match="OpenCL"):
        a.to_device(f"{backend}:0", zero_copy=True)


def test_zero_copy_is_offered_where_the_api_guarantees_it():
    for backend in ("cuda", "hip", "metal"):
        supported, reason = gnp.zero_copy_support(backend)
        assert supported, f"{backend}: {reason}"


def test_zero_copy_refuses_the_interpreter():
    """The ``python`` backend has no device memory, so it has nothing to bind."""
    supported, reason = gnp.zero_copy_support("python")
    assert not supported
    assert "interpreter" in reason


def test_a_plain_numpy_array_is_not_a_gnp_array():
    """``numpy``'s own constructors return numpy arrays: the gnp ones are in ``gnp``."""
    array = np.zeros(4, dtype=np.float32)
    assert not isinstance(array, gnp.ndarray)
    assert isinstance(gnp.zeros(4, dtype=np.float32), gnp.ndarray)


def test_zero_copy_shares_the_arrays_own_memory(gpu_backend):
    """A zero-copy binding makes the device read and write the array's own host bytes."""
    supported, reason = gnp.zero_copy_support(f"{gpu_backend}:0")
    if not supported:
        pytest.skip(reason)

    a = gnp.arange(64, dtype=np.float32)
    a.to_device(f"{gpu_backend}:0", zero_copy=True, wait=True)
    assert a.is_zero_copy(f"{gpu_backend}:0")
    binding = _bindings(a)[(gpu_backend, pygpgpu.get_device(gpu_backend))]
    assert binding.zero_copy
    assert binding.mapped, "a zero-copy binding must be host memory the device addresses"


def test_a_zero_copy_kernel_writes_straight_into_the_array(gpu_backend):
    """The result of a kernel on a zero-copy array is in the array, with no copy back."""
    supported, reason = gnp.zero_copy_support(f"{gpu_backend}:0")
    if not supported:
        pytest.skip(reason)

    a = gnp.arange(64, dtype=np.float32)
    b = gnp.full(64, 1.0, dtype=np.float32)
    out = gnp.zeros(64, dtype=np.float32)
    for array in (a, b, out):
        array.to_device(f"{gpu_backend}:0", zero_copy=True, wait=True)

    saxpy[gpu_backend](a, b, out, 2.0)

    np.testing.assert_allclose(
        out.tolist(), (np.arange(64, dtype=np.float32) * 2 + 1).tolist())


# ---------- deallocation ----------

def test_release_drops_the_device_copy_but_keeps_the_array(gpu_backend):
    a = gnp.arange(64, dtype=np.float32)
    bump[gpu_backend](a)
    assert _bindings(a)
    a.release()
    assert not a._bindings
    assert float(a[0]) == 1.0, "releasing the device copy must keep the host result"
