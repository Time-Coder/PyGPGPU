"""The pooled-buffer claim: a stale release must not free the current holder's claim.

A pooled buffer is handed out with `use()` and handed back from a **driver callback** (the
read-back completion). The driver decides when that callback fires, and a rapid caller can
have the same buffer handed out again before it does:

    launch 1: use()  -> claim #1
    launch 1 completes; the release callback is queued
    launch 2: use()  -> claim #2   (the buffer was free again)
    the queued callback for claim #1 fires and calls unuse()

Without a serial, that last `unuse()` frees *claim #2* and the buffer is handed out a third
time while launch 2 still owns it. The OpenCL backend already carries the serial for exactly
this reason (`_claim_serial`, and a mismatched release is a no-op); the CUDA and HIP pools
did not, and their `unuse()` raised `RuntimeError("mem obj is not in using")` from a driver
thread instead -- a message printed where nobody reads it, with the claim table left wrong.
"""
from __future__ import annotations

import pytest

from pygpgpu.cuda.oop.mem_object import MemObject as CudaMemObject
from pygpgpu.hip.oop.mem_object import MemObject as HipMemObject


def _bare(cls):
    """One MemObject with its driver calls bypassed: the claim logic is what is under test.

    A concrete subclass satisfies the abstract methods without ever calling the driver.
    """
    import threading

    concrete = type(
        f"_Bare{cls.__name__}",
        (cls,),
        {
            "write": lambda self, *a, **k: None,
            "read": lambda self, *a, **k: None,
            "set_data": lambda self, *a, **k: None,
        },
    )
    obj = concrete.__new__(concrete)
    obj._using = False
    obj._claim_serial = 0
    obj._claim_lock = threading.Lock()
    return obj


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_claim_reports_its_serial(cls):
    """`use()` answers the serial a matching release must carry."""
    obj = _bare(cls)
    first = obj.use()
    assert isinstance(first, int)
    obj.unuse(first)
    second = obj.use()
    assert second != first, "a new claim must not reuse the old serial"


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_stale_release_does_not_free_the_new_claim(cls):
    """The bug: a queued release for claim #1 landing after claim #2 was taken."""
    obj = _bare(cls)
    serial_one = obj.use()
    obj.unuse(serial_one)
    serial_two = obj.use()

    # The callback for the *first* launch fires late.
    obj.unuse(serial_one)

    assert obj.using, "a stale release freed the current holder's claim"
    obj.unuse(serial_two)
    assert not obj.using


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_matching_release_frees_the_claim(cls):
    obj = _bare(cls)
    serial = obj.use()
    obj.unuse(serial)
    assert not obj.using


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_release_without_a_serial_still_frees_an_unclaimed_buffer(cls):
    """The no-argument form keeps working: it is what the synchronous path uses."""
    obj = _bare(cls)
    obj.use()
    obj.unuse()
    assert not obj.using


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_double_claim_is_still_an_error(cls):
    """Two owners of one buffer is a real bug, and it must keep raising."""
    obj = _bare(cls)
    obj.use()
    with pytest.raises(RuntimeError):
        obj.use()


@pytest.mark.parametrize("cls", [CudaMemObject, HipMemObject], ids=["cuda", "hip"])
def test_a_failed_launch_releases_the_claims_it_made(cls):
    """`_release_failed_launch` returns every claim a failed launch still holds.

    Regression: the binding loop had no error bracket, so a launch that raised on
    argument 3 left the buffers claimed for arguments 1-2 held forever -- repeated
    failed launches bled the pool dry (every later `get_buffer` of that size allocated
    a fresh buffer because none was ever handed back).
    """
    import numpy as np

    # The helper is per backend (each carries its own MemObject class, which its
    # isinstance test reads), so the parametrized class picks its own.
    if cls is CudaMemObject:
        from pygpgpu.cuda.oop.kernel._kernel_launch import _release_failed_launch
    else:
        from pygpgpu.hip.oop.kernel._kernel_launch import _release_failed_launch

    held = _bare(cls)
    shared = _bare(cls)
    serial = held.use()
    shared.use()
    bound = {
        # An array argument: (mem_obj, original value, converted value).
        "a": (held, np.zeros(1), np.zeros(1)),
        # A scalar or handle argument carries no buffer.
        "k": (None, 1, 1),
        # The same buffer travelling under a second name: claimed once, so the
        # blanket walk releases it twice and the second release is a no-op.
        "b": (shared, np.zeros(1), np.zeros(1)),
    }
    _release_failed_launch(bound)
    assert not held.using, "the failed launch kept its claim"
    assert not shared.using, "the shared buffer's claim was not returned"
    assert held.claim_serial == serial, "a release must not bump the serial"
