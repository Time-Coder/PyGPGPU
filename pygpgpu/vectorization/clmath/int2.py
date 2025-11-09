from .. import genVec2

import ctypes


class int2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_int32