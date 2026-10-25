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
        """Look up a device by index / name or vendor substring / "name:ordinal"."""
        keyword_lower:str = keyword.lower()

        if keyword_lower in self.__devices_map:
            device:Optional[Device] = self.__devices_map[keyword_lower]
            if device is None:
                raise ValueError(f"Device {keyword} not found")
            return device

        name:str = keyword_lower
        index:int = 0
        if ":" in keyword_lower:
            name, idx_part = keyword_lower.rsplit(":", 1)
            if not idx_part.strip().isdigit():
                raise ValueError(f"invalid device selector '{keyword}' (expected 'name[:index]')")
            index = int(idx_part)

        if name.isdigit():
            i:int = int(name)
            self.__fetch_devices()
            if not (0 <= i < len(self.__devices_list)):
                raise ValueError(f"device index {i} out of range (0..{len(self.__devices_list) - 1})")
            device = self.__devices_list[i]
            self.__devices_map[keyword_lower] = device
            return device

        matches:List[Device] = [
            d for d in self
            if name in d.name.lower() or name in getattr(d, "vendor", "").lower()]
        device = (matches[index] if 0 <= index < len(matches) else None)
        self.__devices_map[keyword_lower] = device

        if device is None:
            available:str = ", ".join(d.name for d in self)
            raise ValueError(f"Device {keyword} not found; available: {available}")

        return device
    
    def __getitem__(self, index:int)->Device:
        self.__fetch_devices()
        return self.__devices_list[index]
    
    def __iter__(self)->Iterator[Device]:
        self.__fetch_devices()
        return iter(self.__devices_list)
    
    def __contains__(self, device:Device)->bool:
        return (device.id in self.__devices_set)
        

class Devices(metaclass=MetaDevices):
    pass