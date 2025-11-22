from .. import genVec4, Flavor

import ctypes


class uint4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 16