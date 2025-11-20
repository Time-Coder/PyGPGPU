from .. import genMat3x3, Flavor

import ctypes


class dmat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_double
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 72
    
dmat3 = dmat3x3