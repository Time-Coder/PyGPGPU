"""A device buffer, and the host copies that fill and drain it.

`MTLBuffer` storage is `StorageModeShared` (a unified-memory mapping the CPU and
the GPU both read), which is what makes a plain `memcpy` the whole copy path: no
staging buffer and no explicit blit encoder is needed. The unified-memory
assumption holds on every Apple Silicon Mac; on an Intel Mac with a discrete
GPU, shared storage is still the default for buffers of this size, and Metal
maps it into the host address space, so the same code is correct there too.

The copy policy mirrors the OpenCL Buffer: `set_data` uploads, `read` downloads,
`bind_data` records a host pointer *without* copying (so an argument the kernel
never touches still has a destination for its read-back), and the kernel's
read/write flags decide which of those a launch performs.
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from ..driver.objc import ObjC
from .event import Event
from .metalobject import MTLObject

if TYPE_CHECKING:
    from ...constants import KernelFlags
    from .context import CommandQueue, Context

# MTLResourceOptions: StorageModeShared is (0 << 4); on macOS the CPU can map it.
MTL_RESOURCE_STORAGE_MODE_SHARED:int = 0


class Buffer(MTLObject):
    """A `MTLBuffer` with the host-copy surface the shared launch code expects."""

    def __init__(self, context:"Context",
                 data:Union[bytes, bytearray, np.ndarray, None]=None,
                 size:int=0, flags:Optional[int]=None)->None:
        self._context:"Context" = context
        self._flags:Optional[int] = flags
        self._kernel_flags:"KernelFlags" = KernelFlags(0)
        self._data:Union[bytes, bytearray, np.ndarray, None] = data
        self._host_ptr:Optional[int] = None
        self._using:bool = False
        self._dirty_on_device:bool = False
        self._dirty_on_host:bool = False
        self._device_dirty_time:float = 0.0

        if data is not None:
            data, size = self._contiguous_with_size(data)
            # newBufferWithBytes copies the bytes in, which is a full upload by
            # itself: the host memory need not stay alive afterwards.
            buffer_id = ObjC.send_ptr(
                context.device.id,
                "newBufferWithBytes:length:options:",
                # `length` and `options` are `NSUInteger` (pointer-sized). `c_ulong`
                # is that size on macOS only, so `c_size_t` is the honest type.
                (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t),
                int(data.ctypes.data), int(size), MTL_RESOURCE_STORAGE_MODE_SHARED)
        else:
            buffer_id = ObjC.send_ptr(
                context.device.id,
                "newBufferWithLength:options:",
                (ctypes.c_size_t, ctypes.c_size_t),
                int(size), MTL_RESOURCE_STORAGE_MODE_SHARED)
        if not buffer_id:
            raise RuntimeError(f"Metal could not allocate a {size}-byte buffer")
        MTLObject.__init__(self, buffer_id)
        self._size:int = int(size)
        self._host_ptr = self._contents()

    # ---- the MemObject surface ----

    @property
    def context(self)->"Context":
        return self._context

    @property
    def data(self)->Union[bytes, bytearray, np.ndarray, None]:
        return self._data

    @property
    def flags(self)->Optional[int]:
        return self._flags

    @property
    def kernel_flags(self)->"KernelFlags":
        return self._kernel_flags

    @kernel_flags.setter
    def kernel_flags(self, flags:"KernelFlags")->None:
        self._kernel_flags = flags

    @property
    def host_ptr(self)->Optional[int]:
        return self._host_ptr

    @property
    def size(self)->int:
        return self._size

    @property
    def using(self)->bool:
        return self._using

    def use(self)->None:
        if self._using:
            raise RuntimeError("mem obj is still in using")
        self._using = True

    def unuse(self)->None:
        if not self._using:
            raise RuntimeError("mem obj is not in using")
        self._using = False

    @property
    def dirty_on_device(self)->bool:
        return self._dirty_on_device

    @dirty_on_device.setter
    def dirty_on_device(self, dirty:bool)->None:
        self._dirty_on_device = dirty
        self._device_dirty_time = (__import__("time").time() if dirty else 0.0)

    @property
    def dirty_on_host(self)->bool:
        return self._dirty_on_host

    @dirty_on_host.setter
    def dirty_on_host(self, dirty:bool)->None:
        self._dirty_on_host = dirty

    @property
    def device_dirty_time(self)->float:
        return self._device_dirty_time

    def __len__(self)->int:
        return self._size

    # ---- copies ----

    @staticmethod
    def _contiguous_with_size(data:object)->Tuple[np.ndarray, int]:
        if isinstance(data, bytes):
            return np.frombuffer(data, dtype=np.uint8), len(data)
        if isinstance(data, bytearray):
            return np.frombuffer(data, dtype=np.uint8), len(data)
        array = np.asarray(data)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return array, int(array.nbytes)

    def _contents(self)->int:
        """The host-visible address of the buffer's storage."""
        return int(ObjC.send(self.id, "contents", (), ctypes.c_void_p))

    def _host_byte_view(self)->bytes:
        return ctypes.string_at(self._host_ptr, self._size)

    def write(self, cmd_queue:"CommandQueue", origin:int=0,
              region:Optional[int]=None, host_ptr:Optional[int]=None,
              after_events:Optional[List[Event]]=None)->Event:
        """Copy `region` bytes from host memory into the buffer."""
        if region is None:
            region = self._size
        source = self._host_ptr if host_ptr is None else host_ptr
        ctypes.memmove(self._host_ptr + int(origin), int(source), int(region))
        return Event()

    def read(self, cmd_queue:"CommandQueue", origin:int=0,
             region:Optional[int]=None, host_ptr:Optional[int]=None,
             after_events:Optional[List[Event]]=None)->Event:
        """Copy `region` bytes from the buffer into host memory."""
        if region is None:
            region = self._size
        target = self._host_ptr if host_ptr is None else host_ptr
        ctypes.memmove(int(target), self._host_ptr + int(origin), int(region))
        return Event()

    def _host_pointer_of(self, data:object)->Tuple[int, np.ndarray]:
        array, size = self._contiguous_with_size(data)
        if size != self._size:
            raise ValueError("data size is not equal to buffer size")
        return int(array.ctypes.data), array

    def bind_data(self, data:Union[bytes, bytearray, np.ndarray])->None:
        """Remember the caller's data **without** copying it to the buffer.

        An argument the kernel never reads nor writes contributes nothing and
        receives nothing, but the read-back still needs a host destination --
        which is what this records, exactly as the OpenCL Buffer does.
        """
        host_ptr, array = self._host_pointer_of(data)
        self._host_ptr = host_ptr
        self._data = array

    def set_data(self, cmd_queue:"CommandQueue",
                 data:Union[bytes, bytearray, np.ndarray],
                 after_events:Optional[List[Event]]=None)->Event:
        """Record the host data and upload it into the buffer."""
        host_ptr, array = self._host_pointer_of(data)
        self._data = array
        device_ptr = self._contents()
        ctypes.memmove(device_ptr, host_ptr, self._size)
        return Event()
