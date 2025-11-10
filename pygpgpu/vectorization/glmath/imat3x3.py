from .. import genMat3x3, Flavor

import ctypes


class imat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_int
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
imat3 = imat3x3