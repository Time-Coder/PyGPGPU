from .. import genMat3x2, Flavor

import ctypes


class umat3x2(genMat3x2):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 24