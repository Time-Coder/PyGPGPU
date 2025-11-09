from .. import genVec3

import ctypes


class short3(genVec3):

    @property
    def dtype(self)->type:
        return ctypes.c_int16
    
    @property
    def align_pow2(self)->bool:
        return True