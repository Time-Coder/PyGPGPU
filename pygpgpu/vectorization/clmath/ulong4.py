from .. import genVec4, Flavor

import ctypes


class ulong4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 32