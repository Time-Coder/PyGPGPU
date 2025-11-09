from .. import genVec4

import ctypes


class int4(genVec4):

    @property
    def dtype(self)->type:
        return ctypes.c_int32