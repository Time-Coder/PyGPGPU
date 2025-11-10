from .. import genMat3x3, Flavor

import ctypes


class umat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
umat3 = umat3x3