from .. import genMat4x4, Flavor

import ctypes


class umat4x4(genMat4x4):

    @property
    def dtype(self)->type:
        return ctypes.c_uint
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 64
    
umat4 = umat4x4