from .. import genMat4x2, Flavor

import ctypes


class mat4x2(genMat4x2):

    @property
    def dtype(self)->type:
        return ctypes.c_float
    
    @property
    def flavor(self)->Flavor:
        return Flavor.GL
    
    def __sizeof__(self)->int:
        return 32