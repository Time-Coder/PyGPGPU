from ctypes import c_int, pointer
from typing import List, Dict, Iterator, Optional, Set

from ..driver import CUDA, CUdevice
from .device import Device
    

class MetaDevices(type):

    __n_devices:int = 0
    __devices_list:List[Device] = []
    __devices_map:Dict[str, Device] = {}
    __devices_set:Set[CUdevice] = set()

    @property
    def n_devices(self)->int:
        if MetaDevices.__n_devices == 0:
            n_devices = c_int(0)
            ptr_n_devices = pointer(n_devices)
            CUDA.cuDeviceGetCount(ptr_n_devices)
            MetaDevices.__n_devices = n_devices.value

        return MetaDevices.__n_devices
    
    def __len__(self)->int:
        return self.n_devices
    
    def __fetch_devices(self):
        if self.__devices_list:
            return
        
        for i in range(self.n_devices):
            device_id = CUdevice()
            ptr_device_id = pointer(device_id)
            CUDA.cuDeviceGet(ptr_device_id, i)
            device = Device(device_id.value)
            self.__devices_list.append(device)
            self.__devices_set.add(device_id.value)
    
    def device(self, keyword:str)->Device:
        keyword:str = keyword.lower()
        device:Optional[Device] = None

        if keyword in self.__devices_map:
            device:Device = self.__devices_map[keyword]
        else:
            if ":" in keyword:
                items = keyword.split(":")
                name = items[0]
                index = int(items[1])
            else:
                name = keyword
                index = 0
                
            for device_ in self:
                if name in device_.name.lower() or name in device_.vendor.lower():
                    device = device_
                    break

            self.__devices_map[keyword] = device

        if device is None:
            raise ValueError(f"Device {keyword} not found")
        
        return device
    
    def __getitem__(self, index:int)->Device:
        return self.device(index)
    
    def __iter__(self)->Iterator[Device]:
        self.__fetch_devices()
        return iter(self.__devices_list)
    
    def __contains__(self, device:Device)->bool:
        return (device.id in self.__devices_set)
        

class Devices(metaclass=MetaDevices):
    pass