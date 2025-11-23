from __future__ import annotations
from ctypes import pointer, sizeof
from typing import override, TYPE_CHECKING, Optional

from ..runtime import (
    CL,
    cl_mem_flags,
    cl_int
)
from .mem_object import MemObject

if TYPE_CHECKING:
    from .context import Context


class pipe:

    def __init__(self, max_packets:int, packet_type:Optional[type]=None, flags:cl_mem_flags=(cl_mem_flags.CL_MEM_READ_WRITE|cl_mem_flags.CL_MEM_HOST_NO_ACCESS)):
        packet_size:int = 0
        if packet_type is not None:
            packet_size:int = sizeof(packet_type)

        self._packet_size = packet_size
        self._max_packets = max_packets
        self._packet_type = packet_type
        self._flags = flags

    @property
    def packet_size(self)->int:
        return self._packet_size
    
    @property
    def max_packets(self)->int:
        return self._max_packets
    
    @property
    def packet_type(self)->type:
        return self._packet_type
    
    @packet_type.setter
    def packet_type(self, packet_type:type)->None:
        packet_size:int = 0
        if packet_type is not None:
            packet_size:int = sizeof(packet_type)

        self._packet_size = packet_size
        self._packet_type = packet_type
    
    @property
    def flags(self)->cl_mem_flags:
        return self._flags


class Pipe(MemObject):

    def __init__(self, context:Context, pipe_:pipe):
        self._pipe:pipe = pipe_
        error_code = cl_int(0)
        pipe_id = CL.clCreatePipe(context.id, self._pipe.flags, self._pipe.packet_size, self._pipe.max_packets, None, pointer(error_code))
        MemObject.__init__(self, context, pipe_id, None, None, self._pipe.packet_size * self._pipe.max_packets, self._pipe.flags)

    @property
    def packet_size(self)->int:
        return self._pipe.packet_size
    
    @property
    def max_packets(self)->int:
        return self._pipe.max_packets
    
    @property
    def packet_type(self)->type:
        return self._pipe.packet_type

    @override
    def write(self):
        raise AttributeError(f"'Pipe' object has no attribute 'write'")

    @override
    def read(self):
        raise AttributeError(f"'Pipe' object has no attribute 'read'")

    @override
    def set_data(self):
        raise AttributeError(f"'Pipe' object has no attribute 'set_data'")