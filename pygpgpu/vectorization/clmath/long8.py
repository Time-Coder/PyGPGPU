from .. import genVec8

import ctypes


class long8(genVec8):

    @property
    def dtype(self)->type:
        return ctypes.c_int64