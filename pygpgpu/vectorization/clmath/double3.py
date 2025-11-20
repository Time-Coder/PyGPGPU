from .. import genVec3, Flavor

import ctypes


class double3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 32