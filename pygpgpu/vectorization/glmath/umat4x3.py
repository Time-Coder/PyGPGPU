from .. import genMat4x3, Flavor

import ctypes


class umat4x3(genMat4x3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL