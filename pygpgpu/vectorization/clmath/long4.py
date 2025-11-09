from .. import genVec4

import ctypes


class long4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_int64