from .. import genVec8, Flavor

import ctypes


class ulong8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 64