from .. import genVec3, Flavor

import ctypes


class long3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_int64
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL