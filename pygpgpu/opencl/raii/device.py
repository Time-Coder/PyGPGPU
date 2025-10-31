from __future__ import annotations
from ctypes import c_size_t, byref, c_char, sizeof, _SimpleCData, LittleEndianStructure
from typing import TYPE_CHECKING, Any, get_args, Dict
import enum

if TYPE_CHECKING:
    from .platform import Platform

from ..runtime.cltypes import (
    cl_device_id,
    cl_uint,
    cl_bitfield,
    cl_device_info
)
from ..runtime import CL, CLInfo


class Device:

    def __init__(self, platform:Platform, device_id:cl_device_id):
        self.__platform:Platform = platform
        self.__id:cl_device_id = device_id
        self.__info:Dict[cl_device_info, Any] = {}

    @property
    def platform(self)->Platform:
        return self.__platform
    
    @property
    def id(self)->cl_device_id:
        return self.__id
    
    def __getattr__(self, name:str)->Any:
        key = None
        if hasattr(cl_device_info, f"CL_DEVICE_{name.upper()}"):
            key = getattr(cl_device_info, f"CL_DEVICE_{name.upper()}")
        elif hasattr(cl_device_info, f"CL_DEVICE_{name.upper()}_KHR"):
            key = getattr(cl_device_info, f"CL_DEVICE_{name.upper()}_KHR")
        elif hasattr(cl_device_info, f"CL_{name.upper()}"):
            key = getattr(cl_device_info, f"CL_{name.upper()}")
        elif hasattr(cl_device_info, f"CL_{name.upper()}_KHR"):
            key = getattr(cl_device_info, f"CL_{name.upper()}_KHR")

        if key is None:
            raise AttributeError(f"'Device' object has no attribute '{name}'")

        if key not in self.__info:
            self.__info[key] = self.__fetch_info(key)

        return self.__info[key]
    
    def __repr__(self)->str:
        return f"Device('{self.name}')"
    
    def __parse_result(self, buffer:bytes, cls:type):
        if issubclass(cls, _SimpleCData):
            return cls.from_buffer_copy(buffer).value
        elif issubclass(cls, LittleEndianStructure):
            return cls.from_buffer_copy(buffer)
        elif issubclass(cls, enum.IntEnum):
            return cls(cl_uint.from_buffer_copy(buffer).value)
        elif issubclass(cls, enum.IntFlag):
            return cls(cl_bitfield.from_buffer_copy(buffer).value)
        elif issubclass(cls, str):
            return buffer.value.decode("utf-8")
        elif issubclass(cls, bytes):
            return buffer.raw
        elif cls.__name__.startswith("List"):
            args = get_args(cls)
            ele_cls = args[0]
            step:int = sizeof(ele_cls)
            n:int = len(buffer) // step
            result = []
            offset:int = 0
            for i in range(n):
                value = self.__parse_result(buffer[offset:offset+step], ele_cls)
                result.append(value)
                offset += step
            return result

    def __fetch_info(self, key:cl_device_info)->Any:
        result_size = c_size_t()
        CL.clGetDeviceInfo(self.__id, key, 0, None, byref(result_size))

        result_bytes = (c_char * result_size.value)()
        CL.clGetDeviceInfo(self.__id, key, result_size, result_bytes, None)

        cls = CLInfo.device_info_types[key]
        return self.__parse_result(result_bytes, cls)