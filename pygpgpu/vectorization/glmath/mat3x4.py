from .. import genMat3x4, Flavor

import ctypes


class mat3x4(genMat3x4):

    @property
    def dtype(self)->type:
        return ctypes.c_float