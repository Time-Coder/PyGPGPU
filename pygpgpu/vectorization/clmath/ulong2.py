from .. import genVec2

import ctypes


class ulong2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint64