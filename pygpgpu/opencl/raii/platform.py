from ctypes import LittleEndianStructure, c_void_p, c_size_t, byref, c_char, sizeof, _SimpleCData
from typing import Any, List, get_args, Dict, Iterator
import enum

from ..runtime import CL, CLInfo
from ..runtime.cltypes import (
    cl_platform_info,
    cl_platform_id,
    cl_device_type,
    cl_uint,
    cl_device_id,
    cl_bitfield,
    cl_bool
)
from ..runtime.clconstantes import IntFlag, IntEnum

from .device import Device


class Platform:

    def __init__(self, id:cl_platform_id):
        self.__id:cl_platform_id = id
        self.__info:Dict[str, Any] = {}

        self.__n_devices:int = 0
        self.__devices_list:List[Device] = []
        self.__devices_map:Dict[cl_device_id, Device] = {}

    @property
    def id(self)->c_void_p:
        return self.__id
    
    @property
    def extensions(self)->List[str]:
        return self.__getattr__("extensions").split(" ")
    
    def __getattr__(self, name:str)->Any:
        if name in self.__info:
            return self.__info[name]

        key = None
        if hasattr(cl_platform_info, f"CL_PLATFORM_{name.upper()}"):
            key = getattr(cl_platform_info, f"CL_PLATFORM_{name.upper()}")
        elif hasattr(cl_platform_info, f"CL_PLATFORM_{name.upper()}_KHR"):
            key = getattr(cl_platform_info, f"CL_PLATFORM_{name.upper()}_KHR")
        elif hasattr(cl_platform_info, f"CL_{name.upper()}"):
            key = getattr(cl_platform_info, f"CL_{name.upper()}")
        elif hasattr(cl_platform_info, f"CL_{name.upper()}_KHR"):
            key = getattr(cl_platform_info, f"CL_{name.upper()}_KHR")

        if key is None:
            raise AttributeError(f"'Platform' object has no attribute '{name}'")

        self.__info[name] = self.__fetch_info(key)

        return self.__info[name]
    
    def __iter__(self)->Iterator[Device]:
        self.__fetch_devices()
        return iter(self.__devices_list)
    
    def __contains__(self, device:Device)->bool:
        return (device.id.value in self.__devices_map)
    
    @property
    def n_devices(self)->int:
        if self.__n_devices == 0:
            n_devices = cl_uint()
            CL.clGetDeviceIDs(self.__id, cl_device_type.CL_DEVICE_TYPE_ALL, 0, None, byref(n_devices))
            self.__n_devices = n_devices.value

        return self.__n_devices

    def __fetch_devices(self):
        if self.__devices_list:
            return
        
        device_ids = (cl_device_id * self.n_devices)()
        CL.clGetDeviceIDs(self.id, cl_device_type.CL_DEVICE_TYPE_ALL, self.n_devices, device_ids, None)
        for device_id in device_ids:
            device = Device(self, cl_device_id(device_id))
            self.__devices_list.append(device)
            self.__devices_map[device_id] = device

    def device(self, index:int)->Device:
        self.__fetch_devices()
        return self.__devices_list[index]

    def __repr__(self)->str:
        return f"Platform('{self.name}')"
    
    def __parse_result(self, buffer:bytes, cls:type):
        if issubclass(cls, _SimpleCData):
            return cls.from_buffer_copy(buffer).value
        elif issubclass(cls, LittleEndianStructure):
            return cls.from_buffer_copy(buffer)
        elif issubclass(cls, IntEnum):
            return cls(cl_uint.from_buffer_copy(buffer).value)
        elif issubclass(cls, IntFlag):
            return cls(cl_bitfield.from_buffer_copy(buffer).value)
        elif issubclass(cls, str):
            return buffer.value.decode("utf-8")
        elif issubclass(cls, bytes):
            return buffer.raw
        elif issubclass(cls, cl_bool):
            return bool(cl_uint.from_buffer_copy(buffer).value)
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

    def __fetch_info(self, key:cl_platform_info)->Any:
        result_size = c_size_t()
        CL.clGetPlatformInfo(self.__id, key, 0, None, byref(result_size))

        result_bytes = (c_char * result_size.value)()
        CL.clGetPlatformInfo(self.__id, key, result_size, result_bytes, None)

        cls = CLInfo.platform_info_types[key]
        return self.__parse_result(result_bytes, cls)