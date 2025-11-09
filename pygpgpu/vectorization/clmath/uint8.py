from .. import genVec8

import ctypes


class uint8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32