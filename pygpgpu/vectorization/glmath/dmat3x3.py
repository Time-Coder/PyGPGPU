from .. import genMat3x3, Flavor

import ctypes


class dmat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
dmat3 = dmat3x3