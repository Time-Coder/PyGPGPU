from .. import genMat3x4, Flavor

import ctypes


class umat3x4(genMat3x4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 48