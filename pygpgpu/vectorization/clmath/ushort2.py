from .. import genVec2

import ctypes


class ushort2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16