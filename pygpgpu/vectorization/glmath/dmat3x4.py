from .. import genMat3x4, Flavor

import ctypes


class dmat3x4(genMat3x4):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL