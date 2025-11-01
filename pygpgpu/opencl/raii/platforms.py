from ctypes import c_uint, pointer
from typing import List, Dict, Iterator

from ..runtime import CL
from ..runtime.cltypes import cl_platform_id
from .platform import Platform
    

class MetaPlatforms(type):

    __n_platforms:int = 0
    __platforms_list:List[Platform] = []
    __platforms_map:Dict[cl_platform_id, Platform] = {}

    @property
    def n_platforms(self)->int:
        if MetaPlatforms.__n_platforms == 0:
            n_platforms = c_uint()
            ptr_n_platforms = pointer(n_platforms)
            CL.clGetPlatformIDs(0, None, ptr_n_platforms)
            MetaPlatforms.__n_platforms = n_platforms.value

        return MetaPlatforms.__n_platforms
    
    def __len__(self)->int:
        return self.n_platforms
    
    def __fetch_platforms(self):
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
    
    def __getitem__(self, index:int)->Platform:
        return self.platform(index)
    
    def __iter__(self)->Iterator[Platform]:
        self.__fetch_platforms()
        return iter(self.__platforms_list)
    
    def __contains__(self, platform:Platform)->bool:
        return (platform.id.value in self.__platforms_map)
        

class Platforms(metaclass=MetaPlatforms):
    pass