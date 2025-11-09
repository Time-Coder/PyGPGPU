from .. import genVec16

import ctypes


class uint16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_uint32