from __future__ import annotations

from ctypes import c_void_p, pointer
import ctypes
from typing import List, Optional, TYPE_CHECKING, Tuple, Union, override
import weakref

import numpy as np

from ..driver import (
    CL,
    cl_mem_flags,
    cl_event
)
from ...constants import KernelFlags
from .command_queue import CommandQueue
from .event import Event

if TYPE_CHECKING:
    from .context import Context

from .mem_object import MemObject


class SVMBuffer(MemObject):
    """Coarse-grain SVM allocation: the pointer is both a host address and a
    device address, so it can be embedded into struct kernel arguments."""

    def __init__(self, context:Context, size:int):
        svm_ptr = CL.clSVMAlloc(context.id, cl_mem_flags.CL_MEM_READ_WRITE, size, 0)
        if not svm_ptr:
            raise RuntimeError("clSVMAlloc failed (device does not support SVM?)")

        if not isinstance(svm_ptr, c_void_p):
            svm_ptr = c_void_p(svm_ptr)

        MemObject.__init__(self, context, svm_ptr, None, None, size, cl_mem_flags.CL_MEM_READ_WRITE)
        # the default finalizer calls clReleaseMemObject, but an SVM pointer must be freed with clSVMFree
        self._finalizer = weakref.finalize(self, SVMBuffer._svm_free, context, self.id)

    @staticmethod
    def _svm_free(context:Context, svm_ptr:c_void_p)->None:
        if svm_ptr:
            CL.clSVMFree(context.id, svm_ptr)

    @override
    @staticmethod
    def _release_func():
        return None

    @override
    def write(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as svm buffer")

        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueSVMMemcpy(cmd_queue.id, False, self.id, host_ptr, region, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueSVMMemcpy)

    @override
    def read(self, cmd_queue:CommandQueue, origin:Union[int, Tuple[int,...]]=0, region:Optional[Union[int, Tuple[int,...]]]=None, host_ptr:Optional[c_void_p]=None, after_events:Optional[List]=None)->Event:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as svm buffer")

        if region is None:
            region = self._size

        if host_ptr is None:
            host_ptr = self._host_ptr

        if after_events is None:
            after_events = []

        events_ptr = Event.events_ptr(after_events)
        event_id = cl_event(0)
        CL.clEnqueueSVMMemcpy(cmd_queue.id, True, host_ptr, self.id, region, len(after_events), events_ptr, pointer(event_id))
        return Event(self.context, event_id, CL.clEnqueueSVMMemcpy)

    @override
    def set_data(self, cmd_queue:CommandQueue, data:Union[bytes, bytearray, np.ndarray], after_events:Optional[List]=None)->Optional[Event]:
        if cmd_queue.context != self.context:
            raise ValueError("cmd_queue should be in the same context as svm buffer")

        if isinstance(data, bytes):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer_copy(data)
        elif isinstance(data, bytearray):
            size = len(data)
            host_ptr = (ctypes.c_ubyte * size).from_buffer(data)
        elif isinstance(data, np.ndarray):
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

            host_ptr = data.ctypes.data_as(c_void_p)
            size = data.nbytes

        if size != self._size:
            raise ValueError("data size is not equal to svm buffer size")

        self._host_ptr = host_ptr
        self._data = data

        event = None
        if self.kernel_flags & KernelFlags.Readed:
            event = self.write(cmd_queue, 0, size, host_ptr, after_events)

        return event

    def __len__(self)->int:
        return self._size
