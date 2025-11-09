from .. import genVec4

import ctypes


class uint4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32