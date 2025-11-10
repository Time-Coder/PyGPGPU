from .. import genVec3, Flavor

import ctypes


class ulong3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL