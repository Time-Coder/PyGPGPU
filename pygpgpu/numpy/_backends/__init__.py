"""The per-backend half of a gnp copy: allocate, transfer, wait, release.

Everything here is about *one* operation on *one* device. ``ndarray.py`` decides which
operation to perform; this module knows how each backend spells it.

Three things are load-bearing:

* **Every wait is explicit and local.** A copy returns the backend's own event and the
  caller registers a completion callback on it; nothing blocks "just in case". The
  places that genuinely depend on a result block through :func:`wait_event`,
  :func:`sync_queue` / :func:`wait_target`.
* **Zero-copy is only offered where the API guarantees it.** Metal's ``StorageModeShared``
  buffer *is* host memory (the code checks that the addresses agree), and CUDA/HIP
  mapped pinned memory is host memory at a device address that
  ``cuMemHostGetDevicePointer``/``hipHostGetDevicePointer`` answers for. OpenCL's
  ``CL_MEM_USE_HOST_PTR`` is explicitly *not* that: the runtime is free to keep a private
  copy, so a request there raises instead of handing back an array whose writes might
  never reach the device.
* **A refusal names the backend and the missing capability.** A capability the runtime
  does not have must fail here, where the reason is known, rather than as a fault on the
  device later."""
# This module is the facade of a split: the host-memory registration, the allocation,
# the copies and the zero-copy support each moved to the module named after them,
# and every name it used to publish stays importable from here.
from __future__ import annotations


from ._backends_pinning import (
    _HOST_REGISTER_DEVICEMAP,
    _HOST_REGISTER_PORTABLE,
    _PAGE_SIZE,
    _aligned_range,
    _driver_register,
    _driver_unregister,
    _lock,
    _pin_range,
    _register_range,
    _registered_ranges,
    _registration_lock,
    _unregister_range,
    page_locked,
)
from ._backends_alloc import (
    _allocate_zero_copy,
    _allocate_zero_copy_cuda,
    _allocate_zero_copy_hip,
    _allocate_zero_copy_metal,
    _create_buffer,
    _replace_free_with_unregister,
    _safe_release,
    allocate,
    allocate_image,
)
from ._backends_copy import (
    _require_entry_point,
    device_pointer,
    download,
    error_code_ok,
    event_callback,
    is_host_mapped,
    release,
    settle,
    sync_queue,
    upload,
    wait_event,
    wait_target,
)
from ._backends_zerocopy import (
    zero_copy_backend_support,
    zero_copy_support,
    zero_copy_supported,
)
