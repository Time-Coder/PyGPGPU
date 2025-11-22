from .. import genMat3x3, Flavor

import ctypes


class umat3x3(genMat3x3):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 36
    
umat3 = umat3x3