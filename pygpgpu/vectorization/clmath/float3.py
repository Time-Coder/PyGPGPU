from .. import genVec3

import ctypes


class float3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def align_pow2(self)->bool:
        return True