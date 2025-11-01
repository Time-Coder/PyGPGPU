from __future__ import annotations
from ctypes import c_size_t, byref, c_char
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .platform import Platform

from ..runtime.cltypes import (
    cl_device_id,
    cl_device_info,
)
from ..runtime import CL, CLInfo

from .common import parse_result


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
    
    @property
    def extensions(self)->List[str]:
        return self.__getattr__("extensions").split(" ")
    
    def __getattr__(self, name:str)->Any:
        if name in self.__info:
            return self.__info[name]

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

        self.__info[name] = self.__fetch_info(key)

        return self.__info[name]
    
    def __repr__(self)->str:
        return f"Device('{self.name}')"

    def __fetch_info(self, key:cl_device_info)->Any:
        result_size = c_size_t()
        CL.clGetDeviceInfo(self.__id, key, 0, None, byref(result_size))

        result_bytes = (c_char * result_size.value)()
        CL.clGetDeviceInfo(self.__id, key, result_size, result_bytes, None)

        cls = CLInfo.device_info_types[key]
        return parse_result(result_bytes, cls)