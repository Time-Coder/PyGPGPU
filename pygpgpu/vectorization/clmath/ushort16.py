from .. import genVec16

import ctypes


class ushort16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_uint16