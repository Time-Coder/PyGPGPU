from ctypes import c_uint, pointer
import threading
from typing import List, Dict, Iterator, Optional

from ..driver import CL, cl_platform_id
from .platform import Platform
from .device import Device
    

class MetaPlatforms(type):

    __n_platforms:int = 0
    __platforms_list:List[Platform] = []
    __platforms_map:Dict[cl_platform_id, Platform] = {}
    __devices_map:Dict[str, Device] = {}
    # The platform list is built on first use, and every thread that reaches a kernel
    # launch passes through here. Two of them used to run the fetch at once: both saw the
    # empty list, both called `clGetPlatformIDs`, and both appended to the same list --
    # which duplicated every platform and could hand out a half-built `Platform` object.
    # One lock per step makes each fetch a single critical section.
    __fetch_lock = threading.RLock()

    @property
    def n_platforms(self)->int:
        with MetaPlatforms.__fetch_lock:
            if MetaPlatforms.__n_platforms == 0:
                n_platforms = c_uint()
                ptr_n_platforms = pointer(n_platforms)
                CL.clGetPlatformIDs(0, None, ptr_n_platforms)
                MetaPlatforms.__n_platforms = n_platforms.value

            return MetaPlatforms.__n_platforms
    
    def __len__(self)->int:
        return self.n_platforms
    
    def __fetch_platforms(self):
        with MetaPlatforms.__fetch_lock:
            if self.__platforms_list:
                return

            platform_ids = (cl_platform_id * self.n_platforms)()
            CL.clGetPlatformIDs(self.n_platforms, platform_ids, None)
            for platform_id in platform_ids:
                platform = Platform(cl_platform_id(platform_id))
                self.__platforms_list.append(platform)
                self.__platforms_map[platform_id] = platform

    def platform(self, index:int)->Platform:
        self.__fetch_platforms()
        return self.__platforms_list[index]
    
    def device(self, keyword:str)->Device:
        # A Device instance answers for itself: the decorator's resolver hands one over
        # (see kernel_import), and a name lookup cannot run on it.
        if isinstance(keyword, Device):
            return keyword
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
                
            for platform in self:
                if name in platform.name.lower() or name in platform.vendor.lower():
                    device:Device = platform.devices[index]
                    break

                for d in platform.devices:
                    if name in d.name.lower() or name in d.vendor.lower():
                        device = d
                        break

            self.__devices_map[keyword] = device

        if device is None:
            raise ValueError(f"Device {keyword} not found")
        
        return device
    
    def __getitem__(self, index:int)->Platform:
        return self.platform(index)
    
    def __iter__(self)->Iterator[Platform]:
        self.__fetch_platforms()
        return iter(self.__platforms_list)
    
    def __contains__(self, platform:Platform)->bool:
        return (platform.id.value in self.__platforms_map)
        

class Platforms(metaclass=MetaPlatforms):
    pass