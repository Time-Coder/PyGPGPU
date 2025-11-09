from .. import genVec3

import ctypes


class uint3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32
    
    @property
    def align_pow2(self)->bool:
        return True