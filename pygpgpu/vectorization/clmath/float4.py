from .. import genVec4

import ctypes


class float4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_float