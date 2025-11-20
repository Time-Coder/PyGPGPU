from .. import genMat4x4, Flavor

import ctypes


class mat4x4(genMat4x4):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 64
    
mat4 = mat4x4