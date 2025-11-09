from .. import genVec2

import ctypes


class uchar2(genVec2):

    @property
    def dtype(self)->type:
        return ctypes.c_ubyte