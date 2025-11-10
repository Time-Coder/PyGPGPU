from .. import genMat3x2, Flavor

import ctypes


class dmat3x2(genMat3x2):

    @property
    def dtype(self)->type:
        return ctypes.c_double