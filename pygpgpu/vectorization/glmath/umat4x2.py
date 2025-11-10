from .. import genMat4x2, Flavor

import ctypes


class umat4x2(genMat4x2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL