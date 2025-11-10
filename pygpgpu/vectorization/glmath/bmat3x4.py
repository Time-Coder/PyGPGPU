from .. import genMat3x4, Flavor

import ctypes


class bmat3x4(genMat3x4):

    @property
    def dtype(self)->type:
        return ctypes.c_bool