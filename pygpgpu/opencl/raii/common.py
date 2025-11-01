from typing import get_args
from ctypes import sizeof, _SimpleCData, LittleEndianStructure

from ..runtime.clconstantes import IntEnum, IntFlag
from ..runtime.cltypes import cl_bool, cl_uint


def parse_result(buffer:bytes, cls:type):
    if cls.__name__.startswith("List"):
        args = get_args(cls)
        ele_cls = args[0]
        if issubclass(ele_cls, (IntFlag, IntEnum)):
            step:int = ele_cls.size()
        else:
            step:int = sizeof(ele_cls)
        n:int = len(buffer) // step
        result = []
        offset:int = 0
        for i in range(n):
            value = parse_result(buffer[offset:offset+step], ele_cls)
            result.append(value)
            offset += step
        return result

    if issubclass(cls, _SimpleCData):
        return cls.from_buffer_copy(buffer).value
    elif issubclass(cls, LittleEndianStructure):
        return cls.from_buffer_copy(buffer)
    elif issubclass(cls, IntEnum):
        return cls(cls.dtype().from_buffer_copy(buffer).value)
    elif issubclass(cls, IntFlag):
        return cls(cls.dtype().from_buffer_copy(buffer).value)
    elif issubclass(cls, str):
        return buffer.value.decode("utf-8")
    elif issubclass(cls, bytes):
        return buffer.raw
    elif issubclass(cls, cl_bool):
        return bool(cl_uint.from_buffer_copy(buffer).value)