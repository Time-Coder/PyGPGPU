from .. import genVec3, Flavor

import ctypes


class uint3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 16