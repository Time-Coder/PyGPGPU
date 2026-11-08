# The context, memory and handle plumbing every wrapper shares.
from __future__ import annotations
from ctypes import byref, c_float, c_size_t, c_void_p
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from ....hip.driver import HIP, CUdeviceptr
from ..hipblas import HIPBLAS



def ensure_context()->None:
    """Make a HIP context current if none is.

    A hipBLAS handle belongs to the context that was current when it was created, so a
    call with no context is answered with an error the user cannot act on. The context
    is the backend's own (the ``oop`` layer's), so the buffers a caller builds with the
    driver are the ones the library reads.

    A context that was destroyed while current is still reported as current by the
    runtime, and every call against it then fails; so "current" alone is not enough and
    the device is asked for its name as the liveness test.
    """
    from ....hip.oop.devices import Devices

    device = Devices[0]
    context = device.default_context
    if getattr(context, "id", None) is None:
        context.make_current()
        return
    try:
        context.make_current()
        # A dead context is still "current": ask the device to prove it is alive.
        device.name
    except Exception:
        context.make_current()


def allocate(nbytes:int)->CUdeviceptr:
    """Allocate device memory and return its address."""
    ensure_context()
    pointer = CUdeviceptr(0)
    HIP.hipMalloc(byref(pointer), c_size_t(nbytes))
    return pointer


def free(pointer:CUdeviceptr)->None:
    """Release device memory allocated by :func:`allocate`."""
    HIP.hipFree(pointer)


def to_device(array:ArrayLike)->CUdeviceptr:
    """Copy a host array to a fresh device buffer and return its address."""
    host = np.ascontiguousarray(array)
    pointer = allocate(host.nbytes)
    HIP.hipMemcpyHtoD(pointer, c_void_p(host.ctypes.data), c_size_t(host.nbytes))
    return pointer


def to_host(pointer:CUdeviceptr, array:np.ndarray)->np.ndarray:
    """Copy a device buffer into a host array, and return the array."""
    target = np.ascontiguousarray(array)
    HIP.hipMemcpyDtoH(c_void_p(target.ctypes.data), pointer, c_size_t(target.nbytes))
    return target


def synchronize()->None:
    """Wait until the device has finished everything that was queued on it."""
    ensure_context()
    HIP.hipDeviceSynchronize()


def create_handle()->c_void_p:
    """Create a hipBLAS handle, in the context that is current."""
    ensure_context()
    handle = c_void_p()
    HIPBLAS.hipblasCreate(byref(handle))
    return handle


def destroy_handle(handle:c_void_p)->None:
    """Release a hipBLAS handle."""
    HIPBLAS.hipblasDestroy(handle)


def set_stream(handle:c_void_p, stream:object)->None:
    """Order a handle's work on a stream.

    The stream is the HIP driver's stream object or its raw id.
    """
    stream_id = getattr(stream, "id", stream)
    HIPBLAS.hipblasSetStream(handle, _stream_pointer(stream_id))


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


def _element_type(dtype:np.dtype)->Optional[type]:
    """The ctypes scalar type of a numpy dtype, or None when it is not supported."""
    return {np.dtype(np.float32): c_float, np.dtype(np.float64): None}.get(dtype)


def _is_single(dtype:np.dtype)->bool:
    return dtype == np.dtype(np.float32)


def _is_double(dtype:np.dtype)->bool:
    return dtype == np.dtype(np.float64)


def _device_pointer(pointer:CUdeviceptr, target:type)->object:
    """A device address as the typed pointer an entry point wants."""
    return _cast(c_void_p(int(pointer.value)), target)


def _cast(value:object, target:type)->object:
    from ctypes import cast

    return cast(value, target)


def _scalar_pointer(value:float, dtype:np.dtype)->Tuple[object, object]:
    """A host scalar as the pointer a level-1 call reads it through, and its keeper."""
    holder = c_float(value) if _is_single(dtype) else None
    if holder is None:
        from ctypes import c_double

        holder = c_double(value)
    return byref(holder), holder


def _void(pointer:CUdeviceptr)->c_void_p:
    """A device address as the void pointer the FFT entry points take.

    They are declared `hipfftReal*` / `hipfftComplex*`, but ctypes' `c_void_p` is the
    same pointer and accepts the address without the caller naming the element type --
    which matters because the complex buffers are not a float array ctypes could type.
    """
    return c_void_p(int(pointer.value))
