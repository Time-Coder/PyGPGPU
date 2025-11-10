from .. import genMat4x4, Flavor

import ctypes


class dmat4x4(genMat4x4):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
dmat4 = dmat4x4