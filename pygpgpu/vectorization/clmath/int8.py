from .. import genVec8

import ctypes


class int8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_int32