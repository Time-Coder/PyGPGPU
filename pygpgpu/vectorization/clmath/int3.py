from .. import genVec3, Flavor

import ctypes


class int3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_int32
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL