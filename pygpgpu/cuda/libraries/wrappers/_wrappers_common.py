# The context, memory and handle plumbing every wrapper shares.
from __future__ import annotations
from ctypes import byref, c_float, c_double, c_void_p
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from ....cuda.driver import CUDA, CUcontext, CUdeviceptr
from ..cublas import (
    CUBLAS,
    cublasHandle_t,
)
from ..cusparse import (
    cudaDataType_t,
)



# The element types the wrappers know how to move.
_FLOAT32 = np.dtype(np.float32)
_FLOAT64 = np.dtype(np.float64)


def _device_pointer(value:int)->c_void_p:
    """A device address as the pointer type the library entry points want."""
    return c_void_p(int(value))


def _element_type(dtype:np.dtype)->Optional[type]:
    """The ctypes scalar type of a numpy dtype, or None when it is not supported."""
    if dtype == _FLOAT32:
        return c_float
    if dtype == _FLOAT64:
        return c_double
    return None


def _data_type(dtype:np.dtype)->Optional[int]:
    """The CUDA data type of a numpy dtype, or None when it is not supported."""
    if dtype == _FLOAT32:
        return cudaDataType_t.CUDA_R_32F
    if dtype == _FLOAT64:
        return cudaDataType_t.CUDA_R_64F
    return None


def _as_matrix(array:ArrayLike)->np.ndarray:
    """A 2-D, C-contiguous view of an array, or a refusal naming what is wrong."""
    matrix = np.asarray(array)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {matrix.ndim} dimensions")
    if not matrix.flags["C_CONTIGUOUS"]:
        matrix = np.ascontiguousarray(matrix)
    return matrix


def ensure_context()->None:
    """Make a CUDA context current, if none is usable.

    A library handle belongs to the context that was current when it was created,
    and an allocation needs a context at all, so every wrapper makes sure one is
    current first. Without this the first call of a program that has not set a
    context up fails with ``CUDA_ERROR_INVALID_CONTEXT``, whose own message does
    not say what to do about it.

    The backend's default context for the first device is used. It is the context
    the rest of the CUDA backend uses, so a library buffer and a kernel buffer live
    in the same context and can be mixed.
    """
    context_id = CUcontext()
    CUDA.cuCtxGetCurrent(byref(context_id))
    if context_id.value:
        # A context is current. It may still be a DEAD one -- a context that was
        # destroyed while it was current is still reported as current, and every
        # call against it fails with INVALID_CONTEXT -- so it is probed before it
        # is trusted.
        if _context_is_usable():
            return

    from ....cuda.oop.devices import Devices

    devices = list(Devices)
    if not devices:
        raise RuntimeError(
            "no CUDA device is available: the CUDA libraries need one, and none "
            "was found on this machine")

    devices[0].default_context.make_current()


def _context_is_usable()->bool:
    """Whether the current context still works.

    A context that was destroyed while it was current is still reported as the
    current one, and every call against it fails with INVALID_CONTEXT -- so the
    cheapest possible call is used to find out.
    """
    try:
        pointer = CUdeviceptr(0)
        CUDA.cuMemAlloc_v2(byref(pointer), 1)
    except RuntimeError:
        return False

    try:
        CUDA.cuMemFree_v2(pointer)
    except RuntimeError:
        pass

    return True


def allocate(nbytes:int)->c_void_p:
    """Allocate device memory and return its address.

    The allocation goes through the CUDA driver the backend already uses, so the
    pointer is one the rest of the backend can address as well.
    """
    if nbytes <= 0:
        raise ValueError(f"cannot allocate {nbytes} bytes")

    ensure_context()
    device_pointer = CUdeviceptr(0)
    CUDA.cuMemAlloc_v2(byref(device_pointer), nbytes)
    return c_void_p(int(device_pointer.value))


def free(pointer:c_void_p)->None:
    """Release device memory allocated by :func:`allocate`."""
    CUDA.cuMemFree_v2(CUdeviceptr(int(pointer.value)))


def to_device(pointer:c_void_p, array:np.ndarray)->None:
    """Copy a host array to a device buffer."""
    CUDA.cuMemcpyHtoD_v2(CUdeviceptr(int(pointer.value)),
                         array.ctypes.data, array.nbytes)


def to_host(pointer:c_void_p, array:np.ndarray)->np.ndarray:
    """Copy a device buffer into a host array, and return the array."""
    CUDA.cuMemcpyDtoH_v2(array.ctypes.data,
                         CUdeviceptr(int(pointer.value)), array.nbytes)
    return array


def synchronize()->None:
    """Wait until the device has finished everything that was queued on it."""
    CUDA.cuCtxSynchronize()


def create_handle()->cublasHandle_t:
    """Create a cuBLAS handle.

    The handle belongs to the context that is current when it is created, so a
    context is made current first if the caller has not done it.
    """
    ensure_context()
    handle = cublasHandle_t()
    CUBLAS.cublasCreate_v2(byref(handle))
    return handle


def destroy_handle(handle:cublasHandle_t)->None:
    """Release a cuBLAS handle."""
    CUBLAS.cublasDestroy_v2(handle)


def set_stream(handle:cublasHandle_t, stream:object)->None:
    """Order a handle's work on a stream.

    The stream is the CUDA driver's stream object (a ``Stream`` of the backend's
    ``oop`` layer) or its raw id.
    """
    stream_id = getattr(stream, "id", stream)
    CUBLAS.cublasSetStream_v2(handle, _stream_pointer(stream_id))


def _stream_pointer(stream_id:object)->c_void_p:
    """The raw pointer of a stream id, whatever shape that id has."""
    if isinstance(stream_id, c_void_p):
        return stream_id
    if isinstance(stream_id, int):
        return c_void_p(stream_id)
    value = getattr(stream_id, "value", None)
    if value is not None:
        return c_void_p(int(value))
    raise TypeError(f"cannot read a stream id from {type(stream_id).__name__}")
