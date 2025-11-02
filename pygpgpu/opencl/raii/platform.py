from ctypes import byref
from typing import List, Dict, Iterator

from ..runtime import (
    CL, CLInfo, IntEnum,
    cl_platform_id,
    cl_device_type,
    cl_uint,
    cl_device_id,
    cl_platform_info
)

from .clobject import CLObject
from .device import Device


class Platform(CLObject):

    def __init__(self, id:cl_platform_id):
        CLObject.__init__(self, id)
        self.__n_devices:int = 0
        self.__devices_list:List[Device] = []
        self.__devices_map:Dict[cl_device_id, Device] = {}
    
    @property
    def extensions(self)->List[str]:
        return self.__getattr__("extensions").split(" ")
    
    def __iter__(self)->Iterator[Device]:
        self.__fetch_devices()
        return iter(self.__devices_list)
    
    def __contains__(self, device:Device)->bool:
        self.__fetch_devices()
        return (device.id.value in self.__devices_map)
    
    @property
    def n_devices(self)->int:
        if self.__n_devices == 0:
            n_devices = cl_uint()
            CL.clGetDeviceIDs(self._id, cl_device_type.CL_DEVICE_TYPE_ALL, 0, None, byref(n_devices))
            self.__n_devices = n_devices.value

        return self.__n_devices

    def __fetch_devices(self):
        if self.__devices_list:
            return
        
        device_ids = (cl_device_id * self.n_devices)()
        CL.clGetDeviceIDs(self.id, cl_device_type.CL_DEVICE_TYPE_ALL, self.n_devices, device_ids, None)
        for device_id in device_ids:
            device = Device(cl_device_id(device_id), self)
            self.__devices_list.append(device)
            self.__devices_map[device_id] = device

    def device(self, index:int)->Device:
        self.__fetch_devices()
        return self.__devices_list[index]
    
    @property
    def _prefix(self)->str:
        return "CL_PLATFORM"

    @property
    def _get_info_func(self)->CL.Func:
        return CL.clGetPlatformInfo

    @property
    def _info_types_map(self)->Dict[IntEnum, type]:
        return CLInfo.platform_info_types

    @property
    def _info_enum(self)->type:
        return cl_platform_info
