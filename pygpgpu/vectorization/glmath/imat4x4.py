from .. import genMat4x4, Flavor

import ctypes


class imat4x4(genMat4x4):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
imat4 = imat4x4