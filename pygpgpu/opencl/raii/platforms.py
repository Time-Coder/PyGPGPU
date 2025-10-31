from ctypes import c_uint, pointer
from typing import List

from ..runtime import CL
from ..runtime.cltypes import cl_platform_id
from .platform import Platform


class MetaPlatforms(type):

    __n_platforms:int = 0
    __platforms:List[Platform] = []

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
    
    def platform(self, index:int)->Platform:
        if not self.__platforms:
            platform_ids = (cl_platform_id * self.n_platforms)()
            CL.clGetPlatformIDs(self.n_platforms, platform_ids, None)
            for platform_id in platform_ids:
                self.__platforms.append(Platform(platform_id))

        return self.__platforms[index]
    
    def __getitem__(self, index:int)->Platform:
        return self.platform(index)
        

class Platforms(metaclass=MetaPlatforms):
    pass