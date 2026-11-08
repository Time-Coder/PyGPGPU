# Allocating a device buffer, and the zero-copy variants of it.
from __future__ import annotations
from ._backends_pinning import (
    _aligned_range,
    _register_range,
    _unregister_range,
)
from ._backends_zerocopy import zero_copy_supported
import ctypes
from typing import Callable, Optional
from .._devices import Target, context_for



def allocate(target: Target, size: int, zero_copy: bool = False,
             host_ptr: Optional[int] = None) -> object:
    """Create the device object a binding copies through."""
    context = context_for(target)
    context.make_current()
    if zero_copy:
        return _allocate_zero_copy(target, context, size, host_ptr)
    buffer = _create_buffer(target, context, size)
    buffer._gnp_host_mapped = False
    return buffer


def _create_buffer(target: Target, context: object, size: int) -> object:
    """Allocate a device buffer, in whatever spelling this backend's API needs.

    OpenCL's ``clCreateBuffer`` has no default flags -- passing None reaches ctypes as a
    missing value and fails as an ArgumentError -- so it is given the explicit
    read/write flag every other backend defaults to.
    """
    if target.backend == "opencl":
        from ...opencl.driver import cl_mem_flags

        return context.create_buffer(size=size, flags=cl_mem_flags.CL_MEM_READ_WRITE)
    return context.create_buffer(size=size)


def _allocate_zero_copy(target: Target, context: object, size: int,
                        host_ptr: Optional[int]) -> object:
    """Device memory that *is* the host array's own bytes."""
    if host_ptr is None:
        raise ValueError(
            "a zero-copy binding needs the host array it maps, but no host address was "
            "supplied")
    if target.backend == "metal":
        return _allocate_zero_copy_metal(context, size, host_ptr)
    if target.backend == "cuda":
        return _allocate_zero_copy_cuda(context, size, host_ptr)
    if target.backend == "hip":
        return _allocate_zero_copy_hip(context, size, host_ptr)
    supported, reason = zero_copy_supported(target)
    raise NotImplementedError(reason)


def _allocate_zero_copy_metal(context: object, size: int, host_ptr: int) -> object:
    from ...metal.oop.buffer import Buffer

    buffer = Buffer(context, size=size)
    contents = int(buffer._contents())
    if contents == 0:
        raise RuntimeError("Metal returned a shared buffer with no host address")
    # StorageModeShared already is host memory (the CPU writes the GPU's bytes), so an
    # exact address match is what proves the array's own memory is what the device sees.
    buffer._gnp_host_mapped = (contents == int(host_ptr))
    if not buffer._gnp_host_mapped:
        # A Metal buffer of our own still needs the array's bytes; the binding keeps a
        # copy in that case, which the caller asked not to happen.
        raise NotImplementedError(
            "Metal could not hand back a shared buffer at the array's own address, so "
            "zero-copy cannot be guaranteed for this array")
    return buffer


def _allocate_zero_copy_cuda(context: object, size: int, host_ptr: int) -> object:
    from ...cuda.driver import CUDA, CUdeviceptr

    # A registration is made on the *current* context, so this one has to be current
    # before the driver sees the request.
    context.make_current()

    # The device has to be able to map host memory at all; asking first turns a
    # would-be fault into a sentence. `cuDeviceGetAttribute` fills an int through a
    # pointer, and its attribute argument is the enum's numeric value.
    try:
        from ctypes import byref, c_int

        from ...cuda.driver.cutypes import CUdevice_attribute

        can_map = c_int(0)
        CUDA.cuDeviceGetAttribute(byref(can_map),
                                  int(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY),
                                  context.device.id)
        can_map = can_map.value
    except BaseException:
        # A driver too old to answer the question is assumed to map host memory, and the
        # registration itself is the real test.
        can_map = 1
    if can_map != 1:
        raise NotImplementedError(
            f"device '{context.device.name}' reports "
            f"CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY == 0, so it cannot read host memory "
            f"directly; zero-copy is unavailable on it")

    start, registered, offset = _aligned_range(host_ptr, size)
    _register_range("cuda", start, registered)
    device_base = CUdeviceptr(0)
    try:
        CUDA.cuMemHostGetDevicePointer_v2(device_base, ctypes.c_void_p(start), 0)
    except BaseException:
        _unregister_range("cuda", start)
        raise
    device_address = int(device_base.value) + offset

    from ...cuda.oop.buffer import Buffer
    from ...cuda.oop.mem_object import MemObject

    buffer = Buffer.__new__(Buffer)
    MemObject.__init__(buffer, CUdeviceptr(device_address), None, None, size)
    _replace_free_with_unregister(buffer, "cuda", start, _unregister_range)
    return buffer


def _replace_free_with_unregister(buffer: object, backend: str, start: int,
                                  unregister_range: Callable[[str, int], None]) -> None:
    """Make a registered buffer's release unregister its page range, not free its pointer.

    A zero-copy binding's handle is a *device address of host memory*: it was never
    allocated with ``cuMemAlloc``/``hipMalloc``, so the buffer class's own finalizer
    (which frees the pointer) must not run. Its release is re-pointed at the
    registration instead, which is the resource the binding actually holds -- and the
    buffer keeps its class, so a launch treats it exactly like any other buffer.
    """
    buffer._gnp_unregister = lambda: unregister_range(backend, start)
    buffer._gnp_host_mapped = True
    # `weakref.finalize` holds the release callable it was given at construction, so the
    # one that was installed has to be replaced rather than shadowed.
    try:
        buffer._finalizer.detach()
    except BaseException:
        pass
    import weakref

    weakref.finalize(buffer, _safe_release, buffer._gnp_unregister)


def _safe_release(action: Callable[[], None]) -> None:
    try:
        action()
    except BaseException:
        pass


def _allocate_zero_copy_hip(context: object, size: int, host_ptr: int) -> object:
    from ...hip.driver import HIP, CUdeviceptr

    start, registered, offset = _aligned_range(host_ptr, size)
    _register_range("hip", start, registered)
    device_base = CUdeviceptr(0)
    try:
        HIP.hipHostGetDevicePointer(device_base, ctypes.c_void_p(start), 0)
    except BaseException:
        _unregister_range("hip", start)
        raise
    device_address = int(device_base.value) + offset

    from ...hip.oop.buffer import Buffer
    from ...hip.oop.mem_object import MemObject

    buffer = Buffer.__new__(Buffer)
    MemObject.__init__(buffer, CUdeviceptr(device_address), None, None, size)
    _replace_free_with_unregister(buffer, "hip", start, _unregister_range)
    return buffer


def allocate_image(target: Target, queue: object, image_info: object) -> object:
    """Create the backend's image object for an image-typed kernel argument."""
    context = context_for(target)
    context.make_current()
    return context.create_image(image_info)
