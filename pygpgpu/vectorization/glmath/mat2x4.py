from .. import genMat2x4, Flavor

import ctypes


class mat2x4(genMat2x4):

    @property
    def dtype(self)->type:
        return ctypes.c_float