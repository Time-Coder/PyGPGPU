from .. import genMat4x3, Flavor

import ctypes


class dmat4x3(genMat4x3):

    @property
    def dtype(self)->type:
        return ctypes.c_double