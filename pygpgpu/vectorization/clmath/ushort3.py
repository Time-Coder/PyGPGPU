from .. import genVec3

import ctypes


class ushort3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16
    
    @property
    def align_pow2(self)->bool:
        return True