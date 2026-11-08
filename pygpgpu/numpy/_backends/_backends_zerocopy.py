# Which backends can bind a host buffer as a device buffer.
from __future__ import annotations
from ._backends_copy import _require_entry_point
from typing import Tuple
from .._devices import Target



def zero_copy_supported(target: Target) -> Tuple[bool, str]:
    """Whether this backend can bind host memory to the device, and why not."""
    return zero_copy_backend_support(target.backend)


def zero_copy_backend_support(backend: str) -> Tuple[bool, str]:
    """Whether a backend *can* bind host memory, independent of this machine's devices.

    A capability question rather than a device question: ``tests`` and a caller choosing
    a backend both need to know before a device is resolved -- and resolving one can fail
    for an unrelated reason (Metal on a non-Mac, a backend with no devices at all).
    """
    if backend == "metal":
        # Metal's StorageModeShared *is* host memory; the allocation checks the address.
        return True, ""
    if backend == "cuda":
        try:
            _require_entry_point("cuda", "cuMemHostRegister_v2")
            _require_entry_point("cuda", "cuMemHostGetDevicePointer_v2")
        except NotImplementedError as error:
            return False, str(error)
        return True, ""
    if backend == "hip":
        try:
            _require_entry_point("hip", "hipHostRegister")
            _require_entry_point("hip", "hipHostGetDevicePointer")
        except NotImplementedError as error:
            return False, str(error)
        return True, ""
    if backend == "opencl":
        return False, (
            "OpenCL cannot guarantee zero-copy: CL_MEM_USE_HOST_PTR lets the runtime keep "
            "a private copy of the host data, so a kernel write might never reach the "
            "caller's array. Use zero_copy=False (the default) on this backend, or run the "
            "kernel on cuda/hip/metal")
    return False, f"zero-copy is not implemented for the {backend} backend"


# The public spelling of zero_copy_supported, which the array re-exports.
zero_copy_support = zero_copy_supported
