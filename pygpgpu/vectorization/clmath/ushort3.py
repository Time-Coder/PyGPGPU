from .. import genVec3, Flavor

import ctypes


class ushort3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def align_pow2(self)->bool:
        return True
    
    @property
    def flavor(self)->Flavor:
        return Flavor.CL
    
    def __sizeof__(self)->int:
        return 8