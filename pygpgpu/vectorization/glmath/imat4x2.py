from .. import genMat4x2, Flavor

import ctypes


class imat4x2(genMat4x2):

    @property
    def dtype(self)->type:
        return ctypes.c_int