from .. import genVec16

import ctypes


class short16(genVec16):

    @property
    def dtype(self)->type:
        return ctypes.c_int16