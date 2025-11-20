from ctypes import pointer
from typing import override

from ..runtime import (
    CL,
    cl_mem_flags,
    cl_int
)
from .mem_object import MemObject
from .context import Context


class Pipe(MemObject):

    def __init__(self, context:Context, packet_size:int, max_packets:int, flags:cl_mem_flags=cl_mem_flags.CL_MEM_READ_WRITE|cl_mem_flags.CL_MEM_HOST_NO_ACCESS):
        self._context:Context = context
        error_code = cl_int(0)
        CL.clCreatePipe(context.id, flags, packet_size, max_packets, None, pointer(error_code))

    @override
    def write(self):
        raise AttributeError(f"'Pipe' object has no attribute 'write'")

    @override
    def read(self):
        raise AttributeError(f"'Pipe' object has no attribute 'read'")

    @override
    def set_data(self):
        raise AttributeError(f"'Pipe' object has no attribute 'set_data'")