from .. import genVec4

import ctypes


class ulong4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64